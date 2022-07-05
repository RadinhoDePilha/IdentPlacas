import cv2
import pytesseract
import os
from random import randint
import shutil
import sys
from resources.interface import Ui_IdentificacaodePlacas

class Identificador():

    def render(self, file=None):
        if file == None:
            self.cap = cv2.VideoCapture(0)

        else:
            self.cap = cv2.VideoCapture(file)
       
        self.car_cascade = car_cascade = cv2.CascadeClassifier('cascades/haarcascade_plate_number.xml')
        while True:
            ret, self.img = self.cap.read()
            if type(self.img) == type(None):
                break
    
            gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            car_plates = self.car_cascade.detectMultiScale(gray, 1.1, 1)

            for (x,y,w,h) in car_plates:
                self.single_checker(x, y, w, h)
            
            cv2.imshow('video-car', self.img)
            if cv2.waitKey(33) == 27:
                break
        cv2.destroyAllWindows()


    def single_checker(self, x, y, w, h):
            img = self.img
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
            self.plate = img[y: y+h, x:x+w]
            cv2.imshow('Frame', self.plate)
            temp_archive = f'temp/{randint(0, 900)}.jpg' 
            cv2.imwrite(temp_archive, self.plate)
            self.preProcessamentoRoi(temp_archive)
            string = self.reconhecimentoOCR(temp_archive).replace(' ', '')
            if len(string) > 6 and len(string) < 10:
                cv2.imwrite(f'output/{string}.png', self.plate)
            os.remove(temp_archive)

    def preProcessamentoRoi(self, file):
        if file == None:
            img_roi = cv2.imread('output/roi.png')
        else:
            img_roi = cv2.imread(file)

        # Redimensiona a placa em x4
        img = cv2.resize(img_roi, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

        # Converte para escala de cinza
        img = self.plate
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Binariza a img
        _, img = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)
        cv2.imshow('Limiar', img)

        # Desfoque na Imagem
        img = cv2.GaussianBlur(img, (5, 5), 0)

        cv2.imwrite('output/roi-ocr.png', img)

        return img

        
    def reconhecimentoOCR(self, file):
        if file == None:
            img_roi_ocr = cv2.imread('output/roi-ocr.png')
        else:
            img_roi_ocr = cv2.imread(file)
        if img_roi_ocr is None:
            return
        
        config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 6'
        saida = pytesseract.image_to_string(img_roi_ocr, lang='eng', config=config)


        print(saida, type(saida))
        return saida
    
    def clear_output(self):
        try:
            shutil.rmtree('output')
            os.mkdir('output')
        except:
            pass






if __name__ == "__main__":
    idfier = Identificador()
    idfier.render()
    
