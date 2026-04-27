# UAV-assisted LLM Inference Optimization using PPO

## 1. Giới thiệu dự án

Dự án này nghiên cứu bài toán tối ưu hóa hệ thống UAV hỗ trợ cung cấp dịch vụ suy luận mô hình ngôn ngữ lớn (LLM) trong môi trường mạng hạn chế. Trong hệ thống, UAV đóng vai trò như một thiết bị trung gian hoặc relay giữa người dùng và máy chủ biên, giúp cải thiện chất lượng kênh truyền, giảm độ trễ truyền dữ liệu và hỗ trợ quá trình xử lý tác vụ LLM.

Bài toán được mô hình hóa dưới dạng một môi trường học tăng cường, trong đó UAV cần học cách điều chỉnh vị trí, vận tốc bay và cấu hình dịch vụ để cân bằng giữa các mục tiêu:

- Giảm độ trễ truyền và xử lý dữ liệu.
- Giảm chi phí vận hành hệ thống.
- Giữ chất lượng dịch vụ LLM ở mức chấp nhận được.
- Hạn chế vi phạm các ràng buộc về năng lượng, tốc độ bay và độ trễ.
- Tối ưu chính sách điều khiển trong môi trường thay đổi theo thời gian.

Thuật toán chính được sử dụng là **Proximal Policy Optimization (PPO)**, một thuật toán học tăng cường phù hợp với bài toán có không gian hành động liên tục.

---

## 2. Mục tiêu nghiên cứu

Mục tiêu của dự án là xây dựng một mô hình mô phỏng hệ thống UAV-assisted LLM inference và áp dụng học tăng cường để tìm chính sách điều khiển hiệu quả.

Cụ thể, dự án tập trung vào các nội dung sau:

1. Xây dựng môi trường mô phỏng hệ thống UAV hỗ trợ suy luận LLM.
2. Mô hình hóa các yếu tố chính của hệ thống:
   - Vị trí UAV.
   - Vị trí người dùng.
   - Chất lượng kênh truyền.
   - Độ trễ truyền dữ liệu.
   - Độ trễ xử lý tại edge server.
   - Năng lượng tiêu thụ của UAV.
   - Chất lượng đầu ra của mô hình LLM thông qua chỉ số PPL.
3. Thiết kế môi trường học tăng cường theo chuẩn Gymnasium.
4. Huấn luyện thuật toán PPO để tối ưu chính sách điều khiển UAV.
5. Đánh giá kết quả thông qua các chỉ số như reward, latency, PPL, energy consumption, feasible rate và constraint violation.
6. So sánh PPO với một số baseline đơn giản như Random Policy, Hover Policy hoặc Greedy Policy.

---

## 3. Mô hình hệ thống

Hệ thống nghiên cứu gồm các thành phần chính:

### 3.1 Người dùng

Người dùng phát sinh các yêu cầu suy luận LLM. Mỗi yêu cầu có thể bao gồm:

- Số token đầu vào.
- Kích thước prompt.
- Số token đầu ra dự kiến.
- Dung lượng dữ liệu cần truyền.
- Yêu cầu về độ trễ và chất lượng dịch vụ.

Trong mô phỏng, vị trí người dùng có thể được sinh ngẫu nhiên trong một vùng không gian xác định.

### 3.2 UAV

UAV đóng vai trò trung gian hỗ trợ truyền dữ liệu giữa người dùng và hệ thống xử lý biên. UAV có thể thay đổi vị trí theo thời gian để cải thiện chất lượng kênh truyền và giảm độ trễ.

Các đại lượng liên quan đến UAV gồm:

- Tọa độ UAV.
- Vận tốc bay.
- Năng lượng còn lại.
- Công suất tiêu thụ khi bay.
- Khoảng cách giữa UAV và người dùng.
- Khoảng cách giữa UAV và edge server hoặc power beacon nếu có xét đến.

### 3.3 Edge Server

Edge server thực hiện xử lý tác vụ suy luận LLM. Độ trễ xử lý phụ thuộc vào:

- Kích thước đầu vào.
- Số token đầu ra.
- Cấu hình mô hình LLM.
- Số layer hoặc độ sâu mô hình được chọn.
- Tài nguyên tính toán được cấp phát.

### 3.4 Power Beacon hoặc nguồn sạc laser

Trong một số phiên bản mở rộng, hệ thống có xét đến cơ chế sạc năng lượng không dây bằng laser cho UAV. Khi đó, UAV có thể nhận thêm năng lượng từ nguồn phát laser để kéo dài thời gian hoạt động.

Lưu ý: Nếu trong code hiện tại chưa mô hình hóa đầy đủ sạc laser, phần này chỉ nên được mô tả là định hướng hoặc thành phần mô hình dự kiến, không nên khẳng định là đã triển khai đầy đủ.

---

## 4. Môi trường học tăng cường

Bài toán được xây dựng dưới dạng một môi trường học tăng cường theo chuẩn Gymnasium.

Mỗi episode gồm nhiều time slot. Ở mỗi time slot, agent quan sát trạng thái hiện tại của hệ thống, chọn hành động điều khiển, sau đó môi trường cập nhật trạng thái và trả về reward.

Quy trình tổng quát:

1. Khởi tạo vị trí UAV, người dùng và trạng thái năng lượng.
2. Sinh tác vụ LLM cho người dùng.
3. Agent quan sát trạng thái hệ thống.
4. Agent chọn hành động.
5. Môi trường cập nhật vị trí UAV, tính kênh truyền, độ trễ, năng lượng và PPL.
6. Tính reward và penalty.
7. Kiểm tra điều kiện kết thúc episode.
8. Lặp lại cho đến khi hết số time slot hoặc vi phạm điều kiện kết thúc.

---

## 5. Không gian trạng thái

Observation space biểu diễn các thông tin mà PPO sử dụng để ra quyết định. Các giá trị thường được chuẩn hóa về khoảng `[0, 1]` để giúp quá trình học ổn định hơn.

Một cấu trúc trạng thái điển hình gồm:

| Thành phần trạng thái | Ý nghĩa |
|---|---|
| UAV x-position | Tọa độ x của UAV sau chuẩn hóa |
| UAV y-position | Tọa độ y của UAV sau chuẩn hóa |
| UAV energy | Năng lượng còn lại của UAV |
| Average channel gain | Chất lượng kênh trung bình giữa UAV và người dùng |
| Average task size | Kích thước tác vụ trung bình |
| Average latency | Độ trễ hiện tại hoặc trung bình |
| Average PPL | Chỉ số chất lượng mô hình LLM |
| Time slot index | Vị trí hiện tại trong episode |
| Constraint status | Mức độ vi phạm ràng buộc nếu có |

Trong code, observation space có thể được định nghĩa như sau:

```python
self.observation_space = spaces.Box(
    low=0.0,
    high=1.0,
    shape=(obs_dim,),
    dtype=np.float32
)