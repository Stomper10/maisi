# wavelet_module.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DWT3D(nn.Module):
    """3D Haar Discrete Wavelet Transform using strided convolutions."""
    def __init__(self):
        super().__init__()
        l = torch.tensor([1.0, 1.0])  # Low-pass filter
        h = torch.tensor([1.0, -1.0])  # High-pass filter (순서 변경으로 dc 성분 0으로)

        # --- [수정] 3D Outer Product를 위한 필터 reshaping ---
        # 각 차원(Depth, Height, Width)에 대한 필터를 준비
        l_d, h_d = l.view(2, 1, 1), h.view(2, 1, 1)
        l_h, h_h = l.view(1, 2, 1), h.view(1, 2, 1)
        l_w, h_w = l.view(1, 1, 2), h.view(1, 1, 2)
        
        # 8개의 3D 필터 (2x2x2)를 브로드캐스팅을 통해 생성
        f_lll = l_d * l_h * l_w
        f_llh = l_d * l_h * h_w
        f_lhl = l_d * h_h * l_w
        f_lhh = l_d * h_h * h_w
        f_hll = h_d * l_h * l_w
        f_hlh = h_d * l_h * h_w
        f_hhl = h_d * h_h * l_w
        f_hhh = h_d * h_h * h_w
        
        filters = torch.stack( # (out_channels=8, in_channels=1, D=2, H=2, W=2) 형태로 필터 준비
            [f_lll, f_llh, f_lhl, f_lhh, f_hll, f_hlh, f_hhl, f_hhh], dim=0
        ) / (2**1.5) # 정규화        
        self.filters = nn.Parameter(filters.unsqueeze(1), requires_grad=False)

    def forward(self, x):
        # 입력 x: (B, C, D, H, W)
        in_channels = x.shape[1]
        conv_filters = self.filters.repeat(in_channels, 1, 1, 1, 1)
        # groups=in_channels로 설정하여 각 입력 채널이 8개의 필터와 독립적으로 컨볼루션
        return F.conv3d(x, conv_filters, stride=2, padding=0, groups=in_channels)

class IDWT3D(nn.Module):
    """3D Haar Inverse Discrete Wavelet Transform using strided transpose convolutions."""
    def __init__(self):
        super().__init__()
        dwt = DWT3D()
        self.filters = dwt.filters

    def forward(self, x):
        # 입력 x: (B, 8*C, D, H, W)
        in_channels = x.shape[1]
        out_channels = in_channels // 8 # 복원될 채널 수 (C)
        # Transposed 컨볼루션 가중치 (in_channels, out_channels/groups, kD, kH, kW) # (8*C, 1, 2, 2, 2)
        conv_filters = self.filters.repeat(out_channels, 1, 1, 1, 1)
        return F.conv_transpose3d(x, conv_filters, stride=2, padding=0, groups=out_channels)

# test_wavelet.py
# def run_verification_test():
#     """
#     DWT3D와 IDWT3D 모듈의 정확성을 검증합니다.
#     1. 출력 형태(Shape) 검증
#     2. 완벽한 복원(Perfect Reconstruction) 검증
#     3. 에너지 보존(Energy Preservation) 검증
#     """
#     print("--- Wavelet Module Verification Test ---")

#     # 1. 테스트용 모듈 및 데이터 생성
#     dwt = DWT3D()
#     idwt = IDWT3D()
    
#     # 실제 데이터와 동일한 크기의 임의의 텐서 생성
#     # (Batch, Channels, Depth, Height, Width)
#     original_tensor = torch.randn(2, 4, 64, 64, 32)
#     print(f"Original Tensor Shape: {original_tensor.shape}\n")

#     # 2. DWT 적용 및 형태 검증
#     print("--- Test 1: Output Shape Verification ---")
#     wavelet_tensor = dwt(original_tensor)
#     expected_shape = (2, 32, 32, 32, 16)
#     print(f"Wavelet Tensor Shape: {wavelet_tensor.shape}")
    
#     assert wavelet_tensor.shape == expected_shape, "Shape verification FAILED!"
#     print("✅ Output shape is correct.\n")

#     # 3. IDWT 적용 및 완벽한 복원 검증
#     print("--- Test 2: Perfect Reconstruction Test ---")
#     reconstructed_tensor = idwt(wavelet_tensor)
#     print(f"Reconstructed Tensor Shape: {reconstructed_tensor.shape}")

#     # 두 텐서가 거의 같은지 확인 (부동소수점 오차 감안)
#     is_close = torch.allclose(original_tensor, reconstructed_tensor, atol=1e-6)
    
#     # 평균 제곱 오차(MSE) 계산
#     mse = torch.mean((original_tensor - reconstructed_tensor) ** 2).item()
#     print(f"Reconstruction MSE: {mse:.2e}")

#     assert is_close, "Perfect reconstruction FAILED!"
#     print("✅ Perfect reconstruction successful.\n")
    
#     print("--- All tests passed! The implementation is correct. ---")

#     x = torch.randn(2, 3, 64, 64, 64, device='cuda', dtype=torch.float32)
#     dwt, idwt = DWT3D().cuda(), IDWT3D().cuda()

#     y = dwt(x)            # [B, 8*C, 32, 32, 32]
#     xr = idwt(y)          # [B, C, 64, 64, 64]
#     print((x - xr).abs().max().item())  # fp32면 ~1e-6 수준 기대


# if __name__ == "__main__":
#     run_verification_test()