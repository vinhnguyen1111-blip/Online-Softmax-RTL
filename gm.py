import numpy as np

# ==========================================
# CẤU HÌNH ĐỊNH DẠNG FIXED-POINT
# ==========================================
XF_IN = 12   # Q4.12: 4 bit nguyên, 12 bit thập phân cho đầu vào
XF_OUT = 15  # Q1.15: 1 bit nguyên, 15 bit thập phân cho đầu ra
FRAC_MASK = (1 << XF_IN) - 1  # 0xFFF

# Bảng bù 8 đoạn (PWL 8) tối ưu (EXACT_BOUNDS = True)
COMP_PWL_8 = [
    [0,    [[ 1, 2], [ 1, 5], [-1, 8]]],  # Đoạn 0: [0.000, 0.125)
    [35,   [[ 1, 2], [-1, 5], [-1, 7]]],  # Đoạn 1: [0.125, 0.250)
    [107,  [[ 1, 3], [ 1, 5], [-1, 6]]],  # Đoạn 2: [0.250, 0.375)
    [226,  [[ 1, 2], [-1, 3], [-1, 4]]],  # Đoạn 3: [0.375, 0.500)
    [402,  [[-1, 5], [ 1, 6], [-1, 7]]],  # Đoạn 4: [0.500, 0.625)
    [642,  [[-1, 3], [ 1, 6], [-1, 7]]],  # Đoạn 5: [0.625, 0.750)
    [955,  [[-1, 3], [-1, 4], [-1, 5]]],  # Đoạn 6: [0.750, 0.875)
    [1344, [[-1, 2], [-1, 4], [-1, 6]]]   # Đoạn 7: [0.875, 1.000)
]

# ==========================================
# HÀM MÔ PHỎNG HARDWARE LÕI
# ==========================================
def eval_pwl(f_int):
    """Tính lượng bù PWL 8 đoạn bằng dịch bit nguyên."""
    f_int = int(f_int) & FRAC_MASK
    seg = (f_int >> 9) & 0x7
    c0, terms = COMP_PWL_8[seg]
    v = c0
    for sg, k in terms:
        if sg == 1:
            v += (f_int >> k)
        else:
            v -= (f_int >> k)
    return v

def mitchell_exp_pwl(z_fp, is_output=False):
    """Tính 2^z qua xấp xỉ Mitchell có bù PWL 8 đoạn."""
    z_int = z_fp >> XF_IN 
    z_frac = z_fp & FRAC_MASK
    
    v_comp = eval_pwl(z_frac)
    val_base = (1 << XF_IN) + z_frac - v_comp
    shift_amount = abs(z_int)
    
    if is_output:
        return (val_base << (XF_OUT - XF_IN)) >> shift_amount
    else:
        return val_base >> shift_amount

def lzd_mitchell_log(d):
    """Phát hiện Leading Zero để tính log2 (Mitchell log) của tích lũy d."""
    if d == 0:
        return 0
    
    msb_pos = d.bit_length() - 1 
    k_int = msb_pos - XF_IN
    
    if msb_pos >= XF_IN:
        k_frac = (d >> (msb_pos - XF_IN)) & FRAC_MASK
    else:
        k_frac = (d << (XF_IN - msb_pos)) & FRAC_MASK
        
    return (k_int << XF_IN) + k_frac

def lzd_mitchell_log_pwl(d):
    """
    Phát hiện Leading Zero để tính log2 (Mitchell log) của tích lũy d
    Có tích hợp bù sai số 8 đoạn (PWL 8) để tăng độ chính xác.
    """
    if d == 0:
        return 0
    
    # 1. Tìm bit '1' cao nhất (LZD)
    msb_pos = d.bit_length() - 1 
    k_int = msb_pos - XF_IN
    
    # 2. Cắt 12 bit ngay sau bit '1' cao nhất làm phần thập phân k_frac
    if msb_pos >= XF_IN:
        k_frac = (d >> (msb_pos - XF_IN)) & FRAC_MASK
    else:
        k_frac = (d << (XF_IN - msb_pos)) & FRAC_MASK
        
    # 3. TÍNH LƯỢNG BÙ (Tái sử dụng chung bảng COMP_PWL_8 và hàm eval_pwl)
    v_comp = eval_pwl(k_frac)
    
    # 4. CỘNG LƯỢNG BÙ vào phần thập phân
    # Do sai số của Log là thiếu (True Log > Mitchell Log), phải CỘNG thêm v_comp
    k_frac_comp = k_frac + v_comp
        
    # 5. Ghép lại thành K (Q4.12)
    K = (k_int << XF_IN) + k_frac_comp
    
    return K

def online_softmax_hw_sim(x_q412, block_size=32):
    """Mô phỏng Online Softmax xử lý theo khối 32 phần tử"""
    N = len(x_q412)
    if N == 0:
        return []
        
    M_global = 0
    D_global = 0
    is_first_block = True
    
    # PASS 1: Duyệt vector theo từng block để tìm Max và Sum toàn cục
    for i in range(0, N, block_size):
        block = x_q412[i : i + block_size]
        
        # 1. Tìm Max nội bộ của khối hiện tại và lấy phần nguyên
        local_max = max(block)
        M_local = (local_max >> XF_IN) << XF_IN
        
        # 2. Cập nhật Online Max toàn cục và dịch bit bù (Scaling)
        if is_first_block:
            M_global = M_local
            is_first_block = False
            M_new = M_global
        else:
            if M_local > M_global:
                # Tính lượng bit cần dịch phải để scale tổng cũ (không cần bộ nhân)
                shift = (M_local - M_global) >> XF_IN
                D_global >>= shift 
                M_new = M_local
            else:
                M_new = M_global
                
        # 3. Tính tổng lũy thừa cho khối hiện tại theo M_new
        D_local = 0
        for x in block:
            z_i = x - M_new
            D_local += mitchell_exp_pwl(z_i, is_output=False)
            
        # 4. Cộng dồn vào hệ thống
        D_global += D_local
        M_global = M_new
        
    # LOGARIT TRUNG GIAN (Chuyển tiếp)
    # Lấy Log của tổng khối lượng đã cộng dồn
    K = lzd_mitchell_log_pwl(D_global)
    
    # PASS 2: Trừ (M_global + K) và Xuất kết quả cuối cùng
    y_out_q115 = []
    const_sub = M_global + K
    
    for x in x_q412:
        S_i = x - const_sub
        y = mitchell_exp_pwl(S_i, is_output=True)  # Đầu ra định dạng Q1.15
        y_out_q115.append(y)
        
    return y_out_q115

def softmax_hw_sim(x_q412):
    """Mô phỏng Softmax"""
    max_val = max(x_q412)
    M_int = (max_val >> XF_IN) << XF_IN  # Cắt phần thập phân
    
    # PASS 1: Trừ Max nguyên, Đảo Mitchell và Cộng tích lũy
    d_sum = 0
    for x in x_q412:
        z_i = x - M_int
        val = mitchell_exp_pwl(z_i, is_output=False)
        d_sum += val
        
    # LOGARIT TRUNG GIAN (Chuyển tiếp)
    K = lzd_mitchell_log_pwl(d_sum)
    
    # PASS 2: Trừ (M_final + K) và Xuất kết quả
    y_out_q115 = []
    const_sub = M_int + K
    
    for x in x_q412:
        S_i = x - const_sub
        y = mitchell_exp_pwl(S_i, is_output=True)  # Đầu ra định dạng Q1.15
        y_out_q115.append(y)
        
    return y_out_q115

def true_softmax_base2(x_float):
    """Tính chuẩn Softmax lý thuyết (cơ số 2) để so sánh sai số."""
    # Trừ Max để tránh tràn số khi tính hàm mũ
    x_shift = x_float - np.max(x_float)
    exp_val = 2.0 ** x_shift
    return exp_val / np.sum(exp_val)


# ==========================================
# KHỐI KIỂM THỬ VỚI DỮ LIỆU LỚN
# ==========================================
if __name__ == "__main__":
    np.random.seed(15)  # Cố định seed để dễ đối chiếu
    
    # Các kích thước vector cực lớn
    N_list = [1024, 2048, 4096, 8192, 16384, 32768]
    
    print(f"{'Kích thước N':<15} | {'Max Error':<15} | {'Tại x':<12} | {'Mean Error':<15}")
    print("-" * 65)
    
    for N in N_list:
        # Sinh mảng ngẫu nhiên bao gồm cả số âm và dương
        x_float_random = np.random.uniform(-15.0, 15.0, N)
        
        # Chuyển đổi sang Fixed-Point Q4.12 cho mô phỏng phần cứng
        x_q412 = [int(x * (1 << XF_IN)) for x in x_float_random]
        
        # Lấy lại giá trị float chính xác đã bị lượng tử hóa để làm mốc chuẩn (Fair comparison)
        x_float_quantized = np.array(x_q412) / (1 << XF_IN)
        
        # 1. Chạy hàm mô phỏng phần cứng
        y_hw_q115 = online_softmax_hw_sim(x_q412)
        y_hw_float = np.array(y_hw_q115) / (1 << XF_OUT)  # Đưa về Float để so sánh
        
        # 2. Tính Softmax lý thuyết bằng thư viện Numpy
        y_true_float = true_softmax_base2(x_float_quantized)
        
        # 3. Phân tích sai số
        errors = np.abs(y_hw_float - y_true_float)
        
        max_err = np.max(errors)
        mean_err = np.mean(errors)
        
        # Tìm giá trị x tại vị trí có sai số lớn nhất
        idx_max_err = np.argmax(errors)
        x_at_max = x_float_quantized[idx_max_err]
        
        # In kết quả định dạng thẳng cột
        print(f"{N:<15} | {max_err:<15.8f} | {x_at_max:<12.4f} | {mean_err:<15.8f}")