img = cv2.rectangle(img, (output_x1, output_y1),
                            (output_x2, output_y2), (255, 0, 0), 2)
        cv2.imshow("IMG", img)
        cv2.waitKey(0)