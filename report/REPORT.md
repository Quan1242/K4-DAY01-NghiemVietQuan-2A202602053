# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** 9/11/2026

**Runtime Colab:** T4 GPU

**Python / PyTorch / Ultralytics:** Python `3.13.15` / PyTorch `2.11.0+cu128` / Ultralytics `8.4.145`

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không 

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`): `class_id = 468`, `class_name = cab`, `rank = 1`, `score = 0.510915`, `taxonomy_name = ImageNet-1K`.
- Record này mô tả toàn ảnh như thế nào? Mô hình dự đoán toàn bộ ảnh `traffic` có khả năng thuộc lớp `cab` (xe taxi), với score `0.510915`. Đây là dự đoán cho toàn ảnh do sử dụng model yolo11n-cls.pt (Xác định ảnh thuộc lớp nào).
- Ai định nghĩa class list mà checkpoint có thể dự đoán? Class list có sẵn trong taxonomy ImageNet-1K và checkpoint `yolo11n-cls.pt` định nghĩa, thể hiện rõ ở phần đuôi record(`taxonomy_name`).
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy? `class_id` giúp máy xử lý ổn định, `class_name` giúp con người đọc kết quả, còn `taxonomy_name` xác định bộ nhãn đang sử dụng. Nhờ đó tránh nhầm cùng một ID hoặc tên lớp giữa các taxonomy khác nhau.
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì? Guideline cần quy định cách chọn lớp đại diện cho toàn ảnh, chẳng hạn ưu tiên chủ thể chính hoặc chủ thể chiếm phần lớn nội dung. Trường hợp không xác định được chủ thể chính thì sẽ do con người quyết định.
- Vì sao model score không phải ground truth? Score chỉ là mức độ tin cậy của mô hình đối với dự đoán `cab`. Ground truth phải do con người xác định theo guideline và được con người kiểm tra; score cao vẫn có thể xảy ra hiện tượng phân loại sai.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):Record:`person` có `kitchen`: `score = 0.912625`, `bbox_xyxy = [385.33, 69.24, 498.92, 348.92]`, `bbox_width = 113.58`, `bbox_height = 279.68`.
- Diễn giải vị trí box bằng lời: `bbox_xyxy` gồm hai góc của box theo pixel: góc trên bên trái `(x_min, y_min)` và góc dưới bên phải `(x_max, y_max)`. Box dùng để xác định vị trí của một vật trong ảnh.
- So sánh số prediction ở hai threshold: Ở threshold `0.20`, mô hình phát hiện `17` vật thể; ở threshold `0.35` phát hiện `11` vật thể; ở threshold `0.60` phát hiện `6` vật thể. Khi threshold tăng, các prediction có score thấp bị loại bỏ.
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem? Threshold thấp giúp tăng độ bao phủ và giảm nguy cơ bỏ sót object, nhưng tạo nhiều prediction hơn nên reviewer phải kiểm tra nhiều hơn. Threshold cao giảm khối lượng cần kiểm tra nhưng có thể bỏ sót object có score thấp.
- Đề xuất một quy tắc box chặt: Box phải bao trọn phần nhìn thấy của đúng một object, sát biên object, không bao gồm quá nhiều nền hoặc object khác. Tọa độ phải nằm trong ảnh và thỏa mãn `x_min < x_max`, `y_min < y_max`.
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định? Guideline cần quy định có gán nhãn object chỉ nhìn thấy một phần hoặc không, box bám phần nhìn thấy hay ước lượng toàn bộ object. Nếu không chắc chắn về class hoặc ranh giới box, annotator cần đánh dấu để reviewer quyết định.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
	`instance_id = kitchen-001`, `class_name = person`, `score = 0.899318`, có `348` điểm polygon. Một phần `polygon_xy` là `[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0], [441.0, 73.0], [439.0, 73.0], [438.0, 74.0]]`.
- Polygon bổ sung chi tiết gì so với box?
	Polygon bám theo đường biên của người, nên mô tả hình dạng và phần nhìn thấy chi tiết hơn hình chữ nhật bounding box. Box chỉ cho biết vùng bao quanh object.
- `instance_id` dùng để làm gì và không phải loại ID nào?
	`instance_id = kitchen-001` dùng để phân biệt instance người này với các object khác trong cùng ảnh; nó khác với `class_id` và `class_name` là các trường mô tả loại object.
- Đề xuất một quy tắc biên mask:
	Polygon phải bám sát phần nhìn thấy của object, không bao gồm nền hoặc object khác; các điểm phải nằm trong kích thước ảnh và tạo thành vùng kín có ít nhất ba điểm.
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
	Guideline cần quy định mask bám phần nhìn thấy hay ước lượng phần bị che khuất. Nếu không xác định rõ ranh giới do mờ, tiếp xúc hoặc che khuất, annotator cần đánh dấu để reviewer quyết định.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một class cho toàn ảnh, gồm `class_id` và `class_name` | Ảnh có nhiều chủ thể hoặc model chọn sai class | Chọn class đại diện theo guideline | Kiểm tra class có phù hợp với nội dung ảnh không |
| Phát hiện vật thể | Một class và một bounding box `xyxy` cho mỗi object | Bỏ sót object, box quá rộng, sai class hoặc false positive | Gán nhãn từng object và vẽ box sát phần nhìn thấy | Kiểm tra số object, class và vị trí box |
| Instance segmentation | Một class và một polygon cho mỗi instance | Polygon lệch biên, vùng bị che khuất hoặc hai object dính nhau | Vẽ polygon bám biên từng object | Kiểm tra polygon, `instance_id` và vùng mask |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu:
Chỉ thu thập, sử dụng và lưu trữ những hình ảnh, dữ liệu cần thiết cho đúng mục đích và phạm vi công việc được giao
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:
Người phụ trách trực tiếp để được xác nhận và xử lý trước khi tiếp tục.
## 6. Danh sách bằng chứng

- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
