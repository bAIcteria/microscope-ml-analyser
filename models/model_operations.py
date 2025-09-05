from ultralytics import YOLO
import cv2

model = YOLO("models/best_01_09_2025_monika_agatka_wiktoria_e_100_batch_8.pt")

def get_searching_results(img_path, confidence_level=0.5, save_path=None, result_preview_form="id"):
    results = model(img_path, conf=confidence_level)

    if save_path is not None:
        if result_preview_form == "id":
            img = cv2.imread(img_path)

            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                for idx, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box[:4])
                    
                    # Draw bounding box
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Write element ID
                    cv2.putText(img, f"ID:{idx+1}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imwrite(save_path, img)
        else:
            for result in results:
                result.save(filename=save_path)

    return results