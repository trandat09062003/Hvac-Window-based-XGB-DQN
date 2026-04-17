# Hvac-Window-based-XGB-DQN

**Author** : [Xin,Liu](https://www.researchgate.net/profile/Xin-Liu-476), and [Zhonghua,Gou\*](https://www.researchgate.net/profile/Zhonghua-Gou-2). 

**Publication** : Occupant-centric HVAC and window control: A reinforcement learning model for enhancing indoor thermal comfort and energy efficiency. [Building and Environment](https://www.sciencedirect.com/science/article/abs/pii/S0360132324000398) (2024): 111197.

**Institution** : School of Urban Design, Wuhan University, Wuhan, China;

## Abstract

Occupant behavior plays a crucial role in enhancing indoor thermal comfort and achieving energy efficiency by influencing the operational modes of Heating,Ventilation, and Air Conditioning (HVAC) systems as well as windows. However, accurately quantifying the impact of occupant behavior on the indoor environment presents significant challenges in practical applications. 

This study introduces an innovative approach by leveraging the ASHRAE Global Building Occupant Behavior Database and harnessing the power of XGBoost in conjunction with Deep Q Networks (DQN) to construct a reinforcement learning model. This model enables precise prediction of the impact of occupant behavior on the indoor environment at the next time step under varying indoor-outdoor conditions, simultaneously targeting the dual objectives of indoor thermal comfort and energy conservation. 

By applying the XGB-DQN model in sample rooms of four international cities with distinct features, the results demonstrate a significant increase in indoor thermal comfort duration by 24%, accompanied by a 24.7% decrease in air conditioning usage compared to baseline models and actual occupant data. 

This research represents a pioneering effort in applying reinforcement learning techniques to accurately predict occupant behavior's impact on indoor environments, offering valuable insights for intelligent building design and energy management.


## Keywords
- HVAC and window control
- Occupant behavior
- Reinforcement learning
- Thermal comfort
- Energy efficiency

## Diagram
![Program structure of XGB-DQN for HVAC-Window system control](https://ars.els-cdn.com/content/image/1-s2.0-S0360132324000398-gr3_lrg.jpg)

---

## Hướng dẫn vận hành (Vietnam Adaptation)

Dưới đây là các bước để vận hành lại toàn bộ dự án từ dữ liệu thô đến kết quả so sánh cuối cùng cho khí hậu Việt Nam (Hà Nội).

### Giai đoạn 1: Huấn luyện Mô hình XGBoost (Surrogate Model)
Mô hình này giúp dự báo sự thay đổi nhiệt độ trong nhà, đóng vai trò là "máy giả lập" cho RL.
1. Mở file `XGB-DQN.ipynb`.
2. Chạy các Cell đầu tiên (S1, S2) để tiền xử lý dữ liệu từ `Cleaned_data_encode.csv` và huấn luyện mô hình XGBoost.

### Giai đoạn 2: Sinh dữ liệu mô phỏng Hà Nội
Chúng ta sử dụng simulator Sinergym (EnergyPlus) để tạo ra tập dữ liệu kinh nghiệm riêng cho khí hậu nóng ẩm của Hà Nội.
1. Đảm bảo bạn có file thời tiết Hà Nội trong thư mục `weather/`.
2. Chạy lệnh sau trong terminal:
```bash
python3 scripts/generate_sinergym_data.py --weather weather/VNM_NVN_Hanoi-Noi.Bai.Intl.AP.488200_TMYx.2009-2023.epw
```
3. Kết quả: File `Sinergym_Transition_Data.csv` sẽ được tạo ra.

### Giai đoạn 3: Huấn luyện Agent RL (DQN)
Sử dụng dữ liệu Hà Nội vừa sinh ra để dạy cho bộ não DQN cách điều khiển tối ưu.
1. Chạy lệnh huấn luyện (Yêu cầu cài đặt `sinergym` và `tensorflow`):
```bash
PYTHONPATH=$PYTHONPATH:/usr/local/EnergyPlus-25-2-0 python3 scripts/train_rl.py --episodes 5 --steps 1000 --weather weather/VNM_NVN_Hanoi-Noi.Bai.Intl.AP.488200_TMYx.2009-2023.epw
```
2. Kết quả: Trọng số mô hình sẽ được lưu tại `models/dqn_hvac_vietnam.weights.h5`.

### Giai đoạn 4: Đánh giá và So sánh (RL vs Baseline)
Chạy kịch bản đối trọng để xem Agent RL thực sự tiết kiệm được bao nhiêu điện và có thoải mái không.
1. Chạy lệnh đánh giá:
```bash
PYTHONPATH=$PYTHONPATH:/usr/local/EnergyPlus-25-2-0 python3 scripts/evaluate_performance.py --weather weather/VNM_NVN_Hanoi-Noi.Bai.Intl.AP.488200_TMYx.2009-2023.epw
```
2. Kết quả:
   - File `evaluation_results.txt`: Báo cáo chi tiết % tiết kiệm năng lượng.
   - File `performance_comparison.png`: Biểu đồ so sánh trực quan.

---
**Yêu cầu hệ thống:**
- Python 3.12+
- EnergyPlus 25.2.0 (Cài đặt tại `/usr/local/EnergyPlus-25-2-0`)
- Thư viện: `sinergym`, `tensorflow`, `gymnasium`, `pandas`, `xgboost`, `matplotlib`.
