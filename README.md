# UAV-assisted LLM Inference Optimization using PPO

## 1. Giới thiệu

Dự án này nghiên cứu bài toán tối ưu hóa hệ thống UAV hỗ trợ suy luận mô hình ngôn ngữ lớn, trong đó UAV đóng vai trò là thiết bị relay trên không nhằm hỗ trợ truyền dữ liệu giữa người dùng mặt đất và máy chủ biên.

Trong các khu vực có hạ tầng mạng yếu, không ổn định hoặc khó triển khai trạm gốc cố định, người dùng có thể gặp độ trễ cao khi sử dụng các dịch vụ trí tuệ nhân tạo, đặc biệt là các dịch vụ dựa trên mô hình ngôn ngữ lớn. UAV có thể được sử dụng để cải thiện chất lượng kết nối bằng cách bay đến vị trí phù hợp hơn, từ đó cải thiện chất lượng kênh truyền và giảm độ trễ truyền dữ liệu.

Bài toán trong dự án được mô hình hóa dưới dạng một bài toán học tăng cường. Thuật toán Proximal Policy Optimization được sử dụng để học chính sách điều khiển UAV và lựa chọn cấu hình hệ thống nhằm tối ưu hiệu năng tổng thể.

Mục tiêu chính của hệ thống là giảm tổng chi phí vận hành, bao gồm độ trễ, năng lượng tiêu thụ, chất lượng suy luận LLM và mức độ vi phạm các ràng buộc hệ thống.

---

## 2. Mục tiêu của dự án

Dự án hướng tới các mục tiêu chính sau:

- Xây dựng mô hình hệ thống UAV hỗ trợ suy luận LLM.
- Thiết kế môi trường học tăng cường theo chuẩn Gymnasium.
- Mô hình hóa các thành phần chính gồm người dùng, UAV, kênh truyền, edge server, tác vụ LLM và năng lượng UAV.
- Áp dụng thuật toán PPO để học chính sách điều khiển UAV.
- Tối ưu đồng thời nhiều yếu tố như vị trí UAV, độ trễ, chất lượng LLM, năng lượng tiêu thụ và ràng buộc hệ thống.
- Đánh giá kết quả thông qua các chỉ số reward, latency, PPL, energy consumption, feasible rate và constraint violation.
- So sánh PPO với một số chính sách baseline như Random Policy, Hover Policy và Greedy Policy.

---

## 3. Bối cảnh bài toán

Các dịch vụ LLM yêu cầu tài nguyên tính toán lớn và thường cần kết nối ổn định giữa người dùng và hệ thống xử lý. Trong các khu vực có điều kiện mạng hạn chế, việc truyền dữ liệu từ người dùng đến máy chủ biên có thể gặp các vấn đề như:

- Chất lượng kênh truyền thấp.
- Độ trễ truyền dữ liệu cao.
- Kết nối không ổn định.
- Khó đảm bảo chất lượng dịch vụ.
- Khó triển khai hạ tầng mạng cố định.

UAV có thể hỗ trợ bằng cách đóng vai trò relay trên không. Tuy nhiên, UAV cũng có các giới hạn riêng:

- Dung lượng pin có hạn.
- Không thể di chuyển tùy ý với tốc độ quá lớn.
- Cần duy trì trong vùng hoạt động hợp lệ.
- Cần cân bằng giữa di chuyển, truyền thông và năng lượng.
- Cần lựa chọn cấu hình xử lý LLM phù hợp để không làm tăng độ trễ quá mức.

Vì vậy, bài toán cần một phương pháp tối ưu có khả năng ra quyết định tuần tự theo thời gian. Học tăng cường, đặc biệt là PPO, là một hướng tiếp cận phù hợp cho bài toán này.

---

## 4. Mô tả hệ thống

Hệ thống nghiên cứu gồm các thành phần chính:

1. Người dùng mặt đất.
2. UAV relay.
3. Edge server.
4. Tác vụ suy luận LLM.
5. Mô hình kênh truyền.
6. Mô hình năng lượng UAV.
7. Môi trường học tăng cường.

---

## 5. Người dùng mặt đất

Người dùng mặt đất phát sinh các yêu cầu suy luận LLM. Mỗi yêu cầu có thể bao gồm prompt đầu vào và kết quả đầu ra mong muốn.

Trong mô phỏng, mỗi tác vụ LLM có thể được mô tả bằng các đại lượng sau:

| Thành phần | Ý nghĩa |
|---|---|
| Input tokens | Số token đầu vào của prompt |
| Input data size | Kích thước dữ liệu đầu vào |
| Output tokens | Số token đầu ra dự kiến |
| Output data size | Kích thước dữ liệu đầu ra |
| Latency requirement | Yêu cầu về độ trễ |
| Quality requirement | Yêu cầu về chất lượng suy luận |

Vị trí người dùng được sinh ngẫu nhiên trong một vùng không gian xác định. UAV cần học cách điều chỉnh vị trí để phục vụ người dùng hiệu quả hơn.

---

## 6. UAV relay

UAV đóng vai trò relay để hỗ trợ truyền dữ liệu giữa người dùng và edge server. UAV có thể thay đổi vị trí trong không gian nhằm cải thiện chất lượng kênh truyền.

Các trạng thái quan trọng của UAV gồm:

| Thành phần | Ý nghĩa |
|---|---|
| UAV position | Vị trí UAV trong vùng hoạt động |
| UAV velocity | Vận tốc bay của UAV |
| UAV energy | Năng lượng còn lại của UAV |
| Channel gain | Chất lượng kênh truyền |
| Distance to users | Khoảng cách từ UAV đến người dùng |
| Constraint status | Trạng thái vi phạm ràng buộc |

Nhiệm vụ của UAV không chỉ là bay gần người dùng nhất, mà còn phải cân bằng giữa chất lượng truyền thông, độ trễ, năng lượng và ràng buộc hệ thống.

---

## 7. Edge server và suy luận LLM

Edge server thực hiện xử lý các yêu cầu suy luận LLM. Độ trễ xử lý phụ thuộc vào:

- Số token đầu vào.
- Số token đầu ra.
- Cấu hình mô hình LLM.
- Tài nguyên tính toán.
- Độ phức tạp của mô hình.

Trong dự án này, mô hình LLM được mô phỏng ở mức đơn giản. Thay vì chạy một mô hình LLM thật, hệ thống sử dụng các công thức mô phỏng để ước lượng độ trễ xử lý và chất lượng đầu ra.

Một chỉ số được sử dụng để đại diện cho chất lượng LLM là PPL. PPL càng thấp thì chất lượng mô hình càng tốt. Tuy nhiên, để giảm PPL thường cần sử dụng cấu hình mô hình lớn hơn, dẫn đến độ trễ xử lý cao hơn.

Do đó, bài toán có sự đánh đổi giữa:

- Chất lượng LLM.
- Độ trễ xử lý.
- Tài nguyên tính toán.
- Chi phí hệ thống.

---

## 8. Mô hình kênh truyền

Chất lượng kênh truyền phụ thuộc vào khoảng cách giữa UAV và người dùng. Khi UAV ở gần người dùng hơn, kênh truyền thường tốt hơn, tốc độ truyền dữ liệu cao hơn và độ trễ truyền dữ liệu thấp hơn.

Khoảng cách giữa UAV và người dùng có thể được tính như sau:

```python
def compute_distance(uav_pos, user_pos, uav_altitude):
    dx = uav_pos[0] - user_pos[0]
    dy = uav_pos[1] - user_pos[1]
    dz = uav_altitude

    distance = np.sqrt(dx**2 + dy**2 + dz**2)

    return distance
