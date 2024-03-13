import cv2
import numpy as np
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img
def draw_lines(img, lines, color=(0, 255, 0), thickness=3):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
def process_image(img):
    # Gri tonlamaya çevir
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Gürültüyü azaltmak için Gauss filtresi uygula
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Kenarları tespit etmek için Canny kenar dedektörü uygula
    edges = cv2.Canny(blur, 50, 150)

    # İlgilenilen bölgeyi belirler
    imshape = img.shape
    vertices = np.array([[(0, imshape[0]), (450, 320), (550, 320), (imshape[1], imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)

    # Hough dönüşümü kullanarak çizgileri bulur
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 50, np.array([]), minLineLength=50, maxLineGap=100)

    # Şerit değişikliği tespiti yapar
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if y1 > 320:  # Y ekseninde belirli bir noktadan aşağıda bir çizgi bulunuyorsa
                    cv2.putText(img, 'Şerit Değişikliği!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    break
    # Çizgileri orijinal görüntüye ekler
    line_img = np.zeros((imshape[0], imshape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    # İkili görüntü ve çizgili görüntüyü birleştir
    result = cv2.addWeighted(img, 0.8, line_img, 1, 0)
    return result
# Video akışını açar
cap = cv2.VideoCapture("video.mp4")

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        result = process_image(frame)
        cv2.imshow('Lane Detection', result)
    else:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
