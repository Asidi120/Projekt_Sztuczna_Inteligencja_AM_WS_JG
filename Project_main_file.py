import cv2
import pickle
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

with open("pickles/knn-model.pickle", 'rb') as f:
    knn = pickle.load(f)

with open("pickles/face-labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

mozna = 0
nowa_twarz_dodana = False

while True:
    ret, frame = cap.read()
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
    imgTlo = cv2.imread("Zrodla/Modele/rect1.png")
    imgTlo[171:171 + 480, 32:32 + 640] = frame

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_resized = cv2.resize(roi_gray, (100, 100)).flatten()
        predicted_id, confidence = knn._przewiduj(roi_resized)
        print(confidence,"\n")
        if confidence <=9.50:
            name = labels[predicted_id]
            print(f"{name} (confidence(odleglosc euklidesowa): {confidence:.1f})")
            cv2.rectangle(imgTlo, (32 + x, 171 + y), (32 + x + w, 171 + y + h), (0, 255, 0), 2)
            cv2.putText(imgTlo, name, (32 + x, 171 + y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            imgLupa = cv2.imread("Zrodla/Modele/lupa.png")
            imgTlo[310:310 + 258, 790:790 + 256] = imgLupa
            mozna = 0
        else:
            cv2.rectangle(imgTlo, (32 + x, 171 + y), (32 + x + w, 171 + y + h), (0, 165, 255), 2)
            cv2.putText(imgTlo, "Nieznane", (32 + x, 171 + y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
            imgNapis = cv2.imread("Zrodla/Modele/napis123.png")
            imgTlo[310:310 + 256, 790:790 + 257] =imgNapis
            mozna = 1
    cv2.imshow('Rozpoznawanie Twarzy', imgTlo)
    key = cv2.waitKey(20)
    if key & 0xFF == ord('q'):
        break
    elif mozna == 1 and key & 0xFF == ord('n'):
        print("Nowa twarz - podaj imię:")
        nazwafoldera = input("Imię: ").strip()
        folder_path = os.path.join("Images", nazwafoldera)
        if os.path.exists(folder_path):
            print(f"Osoba '{nazwafoldera}' juz istnieje w bazie!")
            komunikat = f"{nazwafoldera} juz istnieje!"
            komunikat_img = frame.copy()
            cv2.rectangle(komunikat_img, (50, 50), (590, 120), (0, 0, 0), -1)
            cv2.putText(komunikat_img, komunikat, (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Rozpoznawanie Twarzy", komunikat_img)
            cv2.waitKey(1500)
        else:
            os.makedirs(folder_path)
            print(f"Tworzenie folderu: {folder_path}")
            komunikaty = [
                "Spojrz w lewo",
                "Spojrz w prawo",
                "Spojrz w gore",
                "Spojrz w dol",
                "Spojrz na wprost",
                "Usmiechnij sie",
                "Zrob powazna mine",
                "Zamknij oczy",
                "Otworz szeroko oczy",
                "Spojrz lekko w bok"
            ]
            cv2.destroyAllWindows()
            zdjęcia_zrobione = 0
            last_capture_time = cv2.getTickCount()
            zdjęcie_interval = 4
            while zdjęcia_zrobione < 10:
                ret, frame = cap.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
                imgTlo = cv2.imread("Zrodla/Modele/rect1.png")
                imgTlo[171:171 + 480, 32:32 + 640] = frame
                cv2.putText(imgTlo, f"Krok {zdjęcia_zrobione + 1}/10:",
                            (750, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                cv2.putText(imgTlo, f"{komunikaty[zdjęcia_zrobione]}",
                            (750, 430), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                current_time = cv2.getTickCount()
                elapsed_time = (current_time - last_capture_time) / cv2.getTickFrequency()

                if elapsed_time >= zdjęcie_interval:
                    for (x, y, w, h) in faces:
                        roi_color = frame[y:y + h, x:x + w]
                        zdjęcie_path = os.path.join(folder_path, f"{zdjęcia_zrobione + 1}.jpg")
                        cv2.imwrite(zdjęcie_path, roi_color)
                        print(f"Zapisano: {zdjęcie_path}")
                        zdjęcia_zrobione += 1
                        last_capture_time = current_time
                        break

                cv2.imshow("Rozpoznawanie Twarzy", imgTlo)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            print("Zapisano 10 zdjęć nowej osoby.")
            nowa_twarz_dodana = True
            break

