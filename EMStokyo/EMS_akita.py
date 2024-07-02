# -*- coding: utf-8 -*-
'''
Created on Thu May  6 09:58:23 2021

************秋田用********************

・ネット接続チェック　ファイルアクセス可能チェック？
→ネット接続しなくてもファイル作成できている。
・カメラの認識がおそすぎる(3分)
→謎。ワーニングは吐いているけど解析できず。
→サーフェス搭載カメラだと早いから、ターミナルで変な処理が入っている？
(カメラIDも通常は搭載の後に2~割り当てられるが0＆1に割り込んでいる)
・PCキャプチャが座標固定でウィンドウを動かさない必要あり。
→試験ウィンドウを認識して相対座標にすれば行けそう…。(時間があれば・・・)

Loと誤認識についての対策

エラーを残す
@author: HOMMA.KAI
'''

# exeファイル作成方法
# 仮想環境を作成し、必要なライブラリをインストールし軽量化
# 以下コマンド
# cd C:\Users\homma.kai\Desktop\EMS_tokyo\EMS_V02
# 対象フォルダへ移動し
# .EMSV02\Scripts\activate.bat　　　　　仮想環境切り替え
# pyinstaller EMS_V05.py --onefil --hidden-import=sklearn.metrics._pairwise_distances_reduction._datasets_pair --hidden-import=sklearn.metrics._pairwise_distances_reduction._middle_term_computer
# exeファイル作成　-Fと-wでコンソールを非表示と1ファイルにまとめる

# 環境作成時は必要最小限のライブラリと下記コマンドでインストーラをインストール
# pip install pyinstaller
# pip install opencv-python #==4.4.0.40
# pip install numpy
# pip install scikit-learn
# pip install tensorflow
# pip install mss

# ------------------------------------------------------------------------
# ライブラリインポートと試験条件の定義
import pickle
import time
from sklearn.neural_network import MLPClassifier
from datetime import datetime
from tkinter import messagebox
import csv
import tkinter
import sys
import numpy as np
import cv2  # versionは4.4.0.40を指定
import os
import mss

import ctypes

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(True)
    DPI = 1
except:
    pass

# import scipy
# import pandas as pd
# import copy
# import sklearn.datasets
# import sklearn.svm


basepath = "C:\\Users\\SHARE.GIJUTSU.2021\\Desktop\\data"
basepath = "data"

# 試験リスト
module = {}

label = [
    ['機種', 'Text'],
    ['試験', 'Button'],
    ['出力', 'Text'],
    ['初期負荷', 'Text'],
    ['アンテナ', 'Button'],
    ['向き', 'Button'],
    ['目量', 'Button'],
    ['メモ', 'Text'],
    ['数字', 'Button']
]


para = [
    ["test"],
    ["EMS低周波", "EMS高周波", "伝導RF"],
    ["6"],
    ["0"],
    ["水平", "垂直"],
    ["前", "右", "後", "左"],
    ["0.1", "0.05", "1"],
    [''],
    ['7seg', 'ドット(それ以外)']
]
with open("lib\\para.pkl", "rb") as f:
    para = pickle.load(f)  # 読み出し


# 7セグ認識パターン　セグのon/offパターンを用意してマッチングさせる
digits_lookup = {
    (1, 1, 1, 1, 1, 1, 0): 0,
    (0, 1, 1, 0, 0, 0, 0): 1,
    (1, 1, 0, 1, 1, 0, 1): 2,
    (1, 1, 1, 1, 0, 0, 1): 3,
    (0, 1, 1, 0, 0, 1, 1): 4,
    (1, 0, 1, 1, 0, 1, 1): 5,
    (1, 0, 1, 1, 1, 1, 1): 6,
    (0, 0, 1, 1, 1, 1, 1): 6,
    (1, 1, 1, 0, 0, 0, 0): 7,
    (1, 1, 1, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 1, 0, 1, 1): 9,
    (1, 1, 1, 0, 0, 1, 1): 9,
    (0, 0, 0, 0, 0, 0, 1): '-'
}

# マウスエリア設定
xmax = 640
ymax = 480
pt = np.array([[0, 0], [xmax, 0], [xmax, ymax], [0, ymax]])  # エリア4隅の情報を入れる
with open("lib\\pt.pkl", "rb") as f:
    pt = pickle.load(f)  # 読み出し
drawing = False
pointa = 0

camNo_main = []
# camNo_sub = []


# ------------------------------------------------------------------------
# GUI
# ボタンの作成
class Button(tkinter.Button):
    def __init__(self, textlist, master):
        tkinter.Button.__init__(self, master, width=18,
                                text=textlist[0], command=self.btn_click)
        self.textlist = textlist

    # ボタンを押すと文字が切り替わる
    def btn_click(self):
        temp = self.textlist[0]
        length = len(self.textlist)
        for i in range(length-1):
            self.textlist[i] = self.textlist[i+1]
        self.textlist[length-1] = temp
        self.configure(text=self.textlist[0])

# テキスト入力欄を作成


class TextBox(tkinter.Entry):
    def __init__(self, textlist, master):
        tkinter.Entry.__init__(self, width=22)
        self.textlist = textlist
        self.delete(0, tkinter.END)
        self.insert(tkinter.END, textlist[0])

# 試験種類変更時に周波数が自動で変更。ボタンを継承
# →周波数は画面キャプチャで読み取れるようになったので、入力は必要なし


class TestButton(Button):
    def btn_click(self):
        super().btn_click()
        global para, tki
        for i in (2, 3):
            temp = module[i].textlist[0]
            length = len(module[i].textlist)
            for j in range(length-1):
                module[i].textlist[j] = module[i].textlist[j+1]
            module[i].textlist[length-1] = temp
            module[i].delete(0, tkinter.END)
            module[i].insert(tkinter.END, module[i].textlist[0])

# 入力完了後、画像認識モードへ切り替える。ボタンを継承


class Action(Button):
    def btn_click(self):
        super().btn_click()
        global para, tki
        count = -1
        with open("lib\\para.pkl", "wb") as f:
            pickle.dump(para, f)  # 保存
        with open("lib\\pt.pkl", "wb") as f:
            pickle.dump(pt, f)  # 保存
        for i, j in label:
            count = count+1
            if j == 'Text':
                para[count][0] = module[count].get()
        if para[0][0] == "":
            messagebox.showerror(label[0][0], "何か入力してください")
            return
        for i in (2, 3):
            try:
                float(para[i][0])
            except ValueError:
                messagebox.showerror(label[i][0], "数値を入力してください。\n単位は不要です。")
                return
        tki.quit()

# ソフト終了。ボタンを継承


class EndAction(Button):
    def btn_click(self):
        super().btn_click()
        try:
            cap.release()
            # subcap.release()
        except:
            pass
        with open("lib\\para.pkl", "wb") as f:
            pickle.dump(para, f)  # 保存
        with open("lib\\pt.pkl", "wb") as f:
            pickle.dump(pt, f)  # 保存
        tki.destroy()
        sys.exit()


class GetCamNum(Button):
    def btn_click(self):
        super().btn_click()
        # if camNo_main[0] == camNo_sub[0]:
        if False:
            messagebox.showerror("エラー", "違うカメラを指定してください")
            return
        else:
            # cv2.destroyAllWindows()
            tki.quit()

# テキストをGUIに表示する


class Label(tkinter.Label):
    def __init__(self, text, master):
        tkinter.Label.__init__(self, text=text)

# ✕クリックで終了した時用。TK()内のprotocol


def on_closing():
    try:
        cap.release()
        # subcap.release()
    except:
        pass
    for i in capset:
        i.release()
    with open("lib\\para.pkl", "wb") as f:
        pickle.dump(para, f)  # 保存
    with open("lib\\pt.pkl", "wb") as f:
        pickle.dump(pt, f)  # 保存
    cv2.destroyAllWindows()
    tki.destroy()
    sys.exit()

# GUI作成。上記のlabelとparaを使用


def TK():
    tki.geometry('330x390')  # 画面サイズの設定 '330x390''660x780'
    tki.title('試験条件入力')  # 画面タイトルの設定
    count = 0
    l = tkinter.Label(tki, text="試験条件を入力してください。ボタンは押すと切り替わります。")
    l.place(x=20*DPI, y=10*DPI)

    for i, j in label:
        l = tkinter.Label(tki, text=i)
        l.place(x=30*DPI, y=(40+30*count)*DPI)
        if j == 'Button':
            module[count] = Button(para[count], master=tki)
            module[count].place(x=110*DPI, y=(40+30*count)*DPI)
        elif j == 'Text':
            module[count] = TextBox(para[count], master=tki)
            module[count].place(x=110*DPI, y=(40+30*count)*DPI)
        count = count+1

    action = Action(["入力完了"], master=tki)
    action.place(x=20*DPI, y=(40+30*count+20)*DPI)

    action = EndAction(["測定終了"], master=tki)
    action.place(x=180*DPI, y=(40+30*count+20)*DPI)

    tki.protocol("WM_DELETE_WINDOW", on_closing)

# カメラID選択のGUI作成


def CameraSet():
    tki.geometry('280x150')  # 画面サイズの設定 '280x150''560x300'
    tki.title('カメラ番号入力')  # 画面タイトルの設定
    l = tkinter.Label(tki, text="ウィンドウを見てカメラ番号を入力してください。\nボタンは押すと切り替わります。")
    l.place(x=20*DPI, y=10*DPI)

    l = tkinter.Label(tki, text="メインカメラ")
    l.place(x=30*DPI, y=(40+30)*DPI)
    Mcam = Button(camNo_main, master=tki)
    Mcam.place(x=110*DPI, y=(40+30)*DPI)
    # l = tkinter.Label(tki, text="サブカメラ(PC)")
    # l.place(x=30, y=40+60)
    # Scam = Button(camNo_sub, master=tki)
    # Scam.place(x=110, y=40+60)

    action = GetCamNum(["入力完了"], master=tki)
    action.place(x=70*DPI, y=(40+60)*DPI)

    tki.protocol("WM_DELETE_WINDOW", on_closing)

# GUI入力完了後に試験条件の読み込みとCSV(ソフト内はpandas)の準備


def getvalue():
    global csvpath, filepath, baseload, meryou, seg, memo
    # 現在日時
    date = datetime.now()
    date = date.strftime('%y%m%d%H%M')
    power = para[2][0]
    baseload = float(para[3][0])
    if para[1][0] == "EMS低周波":
        Class = "MHz"
    elif para[1][0] == "EMS高周波":
        Class = "GHz"
    elif para[1][0] == "伝導RF":
        Class = "RF"
    if para[4][0] == "水平":
        antenna = "H"
    else:
        antenna = "V"
    if para[5][0] == "前":
        angle = "F"
    elif para[5][0] == "後":
        angle = "B"
    elif para[5][0] == "左":
        angle = "L"
    else:
        angle = "R"

    if para[6][0] == "0.1":
        meryou = 10
    elif para[6][0] == "0.05":
        meryou = 100
    else:
        meryou = 1
    memo = para[7][0]
    if para[8][0] == "7seg":
        seg = True
    else:
        seg = False

    filepath = basepath+"\\"+para[0][0]
    filename = Class+date+antenna+angle+power+para[3][0]+".csv"
    csvpath = filepath+"\\"+filename  # 機種名をファイル名にする

    # 回線非接続でも吐き出せそう。念のために残しておく。
    # th=os.environ["TEMP"]
    # temppath=th+"\\error_"+para[0][0]+".csv"#回線非接続時はローカルに保存

# -------------------------------------------------------------------------------------

# マウス操作


def callback(event, x, y, flags, param):
    global pt, drawing, pointa
    distance = [0, 0, 0, 0]

    # 左クリック押下時1番近いマーカーを決定
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        for i in range(4):
            distance[i] = np.linalg.norm([pt[i, 0]-x, pt[i, 1]-y])
        pointa = distance.index(min(distance))

    # マウス移動時、マーカーの座標を移動
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            pt[pointa] = [max(0, min(x, xmax)), max(0, min(y, ymax))]

    # 左クリック離したら移動モード解除
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        pt[pointa] = [max(0, min(x, xmax)), max(0, min(y, ymax))]

# -------------------------------------------------------------------------------------
# 画像処理　対象エリアをメインループから持ってきた後の処理

# 前処理


def preprocessing(img, seg, threshold):
    # グレースケールの画像を2値化(cv2.ADAPTIVE_THRESH_GAUSSIAN_Cを採用)
    # 範囲内のガウス平均値にバイアス与えた値で2値化 cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    # 範囲内の中央値にバイアス与えた値で2値化 cv2.ADAPTIVE_THRESH_MEAN_C
    # 画像全体のヒストグラムの2大ピークからの中央値を閾値に2値化 cv2.threshold(img_psp, 0, 255, cv2.THRESH_OTSU)
    if BkColor == "white":
        # twotone_img = cv2.adaptiveThreshold(img_psp,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        #     cv2.THRESH_BINARY,51,threshold)
        twotone_img = cv2.adaptiveThreshold(img_psp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 99, threshold)
    elif BkColor == "black":
        # twotone_img = cv2.adaptiveThreshold(img_psp,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        #     cv2.THRESH_BINARY,51,0-threshold)
        twotone_img = cv2.adaptiveThreshold(img_psp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 99, 0-threshold)
        twotone_img = cv2.bitwise_not(twotone_img)  # 背景黒の場合は白黒反転

    # 画像端での処理のために余白追加
    area = np.ones((120, 220), dtype='uint8')*255
    area[10:110, 10:210] = twotone_img

    # 中央値フィルタでノイズ除去、3×3のピクセルの中央値フィルタを2段構成
    median1st = cv2.medianBlur(area, 3)
    median = cv2.medianBlur(median1st, 3)
    # median_copy = cv2.medianBlur(median1st,3)  # エリア認識用にもう一つ用意

    # ラベリングして面積の小さいもの(ノイズ・ゴミ等)を削除
    remove_median = area_size_filtering(median)

    # 膨張収縮処理、7segが上下で分離されないように縦方向に膨張させて結合させている
    if seg == True:
        kernel = np.ones((9, 3), np.uint8)  # 膨張収縮用行列
    elif seg == False:
        kernel = np.ones((3, 1), np.uint8)  # 膨張収縮用行列
    opening = cv2.morphologyEx(remove_median, cv2.MORPH_OPEN, kernel)

    # 処理用に追加した余白エリアを削除
    return opening[10:110, 10:210], median[10:110, 10:210]

# ドット判別用にラベリング、面積が一定以上のみの集合とそれ以外の集合に分離する


def area_size_filtering(img):
    # temp_img = img

    # ラベリング処理
    img = cv2.bitwise_not(img)
    label = cv2.connectedComponentsWithStats(img)
    img = cv2.bitwise_not(img)

    # オブジェクト情報を項目別に抽出
    data = np.delete(label[2], 0, 0)
    for i in range(label[0] - 1):
        if data[i][4] > 100:  # 面積でフィルタをかけてる(ここの数字の根拠が甘い)
            # temp_img[label[1] == i + 1] = 255
            img[label[1] == i + 1] = 0
        else:
            img[label[1] == i + 1] = 255

    return img  # 一定以上の大きさの塊だけ残した画像(配列)を返す

# ラベリングで一定面積の物体の外接矩形を保存


def find_digits_positions(img):
    digits_positions = []

    # ラベリング処理
    img = cv2.bitwise_not(img)
    label = cv2.connectedComponentsWithStats(img)

    # オブジェクト情報を項目別に抽出
    data = np.delete(label[2], 0, 0)  # 各ラベルの[X原点,Y原点,X幅,Y幅,面積]

    for i in range(label[0] - 1):
        if data[i][4] < 5000:  # 面積でフィルタをかけてる(4桁以上を想定して200*100/4=5000)
            x0 = data[i][0]
            y0 = data[i][1]
            x1 = data[i][0] + data[i][2]
            y1 = data[i][1] + data[i][3]
            digits_positions.append([[x0, y0], [x1, y1]])

    digits_positions.sort()  # 桁数の大きいものから並べ変え

    return digits_positions  # 数字の左上と右下の座標をリスト化して返す

# 数字認識


def recognize_digits(digits_positions, input_img, seg):

    digits = []
    count = 0
    if seg == True:
        for c in digits_positions[:]:
            x0, y0 = c[0]
            x1, y1 = c[1]
            roi = input_img[y0:y1, x0:x1]
            h, w = roi.shape

            # 1の場合
            if h/w > 4 and x0 != 0:  # h/w値も根拠なし
                digit = 1
                count = count+1
            elif h/w > 1 and 20 < w < 80 and 40 < h:
                count = count+1
                # エリアから7セグ箇所を指定 (xa, ya), (xb, yb)
                segments = [
                    ((w//2, 0), (w//2+1, w//2)),
                    ((w//2, h//4), (w, h//4+1)),
                    ((w//2, h*3//4), (w, h*3//4+1)),
                    ((w//2, h-w//2), (w//2+1, h)),
                    ((0, h*3//4), (w//2, h*3//4+1)),
                    ((0, h//4), (w//2, h//4+1)),
                    ((w//2, h//2-w//4), (w//2+1, h//2+w//4))
                ]
                # エリア内が一定以上塗りつぶされているかで7segのフラグをそれぞれ立てる
                on = [1] * len(segments)
                for (i, ((xa, ya), (xb, yb))) in enumerate(segments):
                    seg_roi = roi[ya:yb, xa:xb]
                    total = cv2.countNonZero(seg_roi)
                    area = (xb - xa) * (yb - ya)
                    if total / max(1, float(area)) > 0.9:  # 閾値は要検討？
                        on[i] = 0
                # フラグ配列がdigits_lookupと一致しているか調べる
                if tuple(on) in digits_lookup.keys():
                    digit = digits_lookup[tuple(on)]
                else:
                    digit = '*'
            elif h/w < 0.75 and count == 0 and w < 60 and 35 < y0 and y1 < 65:
                digit = '-'
                count = count+1
            else:
                digits_positions.remove(c)
                continue

            digits.append(digit)

    elif seg == False:
        for c in digits_positions[:]:

            x0, y0 = c[0]
            x1, y1 = c[1]
            roi = input_img[y0:y1, x0:x1]
            h, w = roi.shape

            if h/w > 1 and 10 < w < 80 and 40 < h:
                count = count+1
                temp = cv2.bitwise_not(roi)
                contours, hierarchy = cv2.findContours(
                    temp, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                # エリアから7セグ箇所を指定 (xa, ya), (xb, yb)
                roi = cv2.resize(roi, dsize=(8, 16))
                roi = np.asarray(roi, dtype=float)
                roi = np.floor(16 - 16 * (roi / 256))
                roi = roi.flatten()
                if len(contours) == 3:
                    digit = int(NN3.predict(roi.reshape(1, -1)))
                elif len(contours) == 1:
                    digit = int(NN2.predict(roi.reshape(1, -1)))
                elif len(contours) == 2:
                    digit = int(NN1.predict(roi.reshape(1, -1)))
                else:
                    digit = '*'
            elif h/w < 0.75 and count == 0 and w < 60 and 35 < y0 and y1 < 65:
                digit = '-'
                count = count+1
            else:
                digits_positions.remove(c)
                continue

            digits.append(digit)

    return digits

# list型の数字を数値型に直す


def cul_digits(digits):
    # 個々の判定結果を合計しint型へ変換
    a = 1
    ans = 0
    if len(digits) == 0:
        ans = "error"
        return ans
    for i in digits:
        if i == '-':
            a = -1
        elif i == '*':  # 認識に失敗した場合はエラーを返す
            ans = "error"
            return ans
        else:
            ans = ans*10+i

    return a*ans/meryou

# 結果を画面に表示する


def disp_process(img, digits_positions, digits, num, result):
    count = 0
    output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for c in digits_positions:
        x0, y0 = c[0]
        x1, y1 = c[1]

        # 画像上に単体の枠と判定結果を表示
        cv2.rectangle(output_img, (x0, y0), (x1, y1), (0, 128, 0), 2)
        cv2.putText(output_img, str(
            num[count]), (x0 + 3, y0 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 128, 0), 2)
        count += 1

    # 画像右上に合計の結果を表示
    cv2.putText(output_img, str(digits), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 128), 2)
    # 画像左下に合計の結果を表示
    cv2.putText(output_img, str(result), (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 128), 2)

    # 測定中かを右上に表示
    # 切断中は赤□
    if state == 0:
        cv2.rectangle(output_img, (185, 5), (195, 15),
                      (0, 0, 255), thickness=-1)
    elif state == 1:
        # 照射待機中は赤〇
        if phase == 0:
            cv2.circle(output_img, (190, 10), 10, (0, 0, 255), thickness=-1)
        # 照射中は緑〇
        elif phase == 1:
            cv2.circle(output_img, (190, 10), 10, (0, 255, 0), thickness=-1)

    return output_img

# -------------------------------------------------------------------------------------
# PCキャプチャ画面から照射中の周波数認識
# テンプレートマッチング→数字認識でも使用しているSVMへ変更


def freq_check(fbox):
    digits_positions = find_digits_positions(fbox)
    digits = []
    count = 0
    x = 0
    for c in digits_positions:

        x0, y0 = c[0]
        x1, y1 = c[1]
        roi = fbox[y0:y1, x0:x1]
        if 10 < (y1-y0) < 80 and 2 < (x1-x0) < 50:
            count += 1
            temp = cv2.bitwise_not(roi)
            contours, hierarchy = cv2.findContours(
                temp, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            # エリアから7セグ箇所を指定 (xa, ya), (xb, yb)
            roi = cv2.resize(roi, dsize=(8, 16))
            roi = np.asarray(roi, dtype=float)
            roi = np.floor(16 - 16 * (roi / 256))
            roi = roi.flatten()

            digit = int(NN.predict(roi.reshape(1, -1)))

            digits.append(digit)
        elif (y1-y0) < 10 and (x1-x0) < 10:
            # ドットを識別
            x = count
        else:
            # 枠線等は除外
            pass
    # print(file)
    # print(digits)
    # num = recognize_digits(digits_positions, fBox, False)
    # cv2.imwrite("lib\\kyousi.png", kyousidata)
    digits = digits[:-1]
    ans = cul_digits(digits)
    ans = 0
    if len(digits) == 0:
        ans = "error"
        return ans
    for i in digits:
        if i == '*':  # 認識に失敗した場合はエラーを返す
            ans = "error"
            return ans
        else:
            ans = ans*10+i
    # x = 3
    try:
        ans = ans / (10**3)
        # ans = ans * meryou/(10**2)
    except:
        ans = "error"
    # print(ans,digits,x)
    return ans


# -------------------------------------------------------------------------------------
# NNの学習
def nn_fitting():
    # SVC学習
    dataset = cv2.imread("lib\\DeepLearningData_.png",
                         cv2.IMREAD_GRAYSCALE)
    # dataset = cv2.imread("lib\\AIdata_00.png", cv2.IMREAD_GRAYSCALE)

    # NN = MLPClassifier(hidden_layer_sizes=(100, 200, 100), activation='relu',
    #                    solver='adam', max_iter=1000)
    # NN.fit(dataset[:, :-1], dataset[:, -1])

    dataset1 = []
    dataset2 = []
    dataset3 = []
    count = 1
    for i in dataset:
        if count % 10 == 0:
            dataset1.append(i)
            dataset3.append(i)
        elif count % 10 == 4:
            dataset1.append(i)
            dataset2.append(i)
        elif count % 10 == 6:
            dataset1.append(i)
        elif count % 10 == 8:
            dataset3.append(i)
            # pass
        elif count % 10 == 9:
            dataset1.append(i)
        else:
            dataset2.append(i)
        count += 1
    dataset1 = np.array(dataset1, dtype='uint8')
    dataset2 = np.array(dataset2, dtype='uint8')
    dataset3 = np.array(dataset3, dtype='uint8')
    # dataset1 = np.delete(dataset, [1,2,3,5,7,8,11,12,13,15,17,18,21,22,23,25,27,28], 0)
    NN1 = MLPClassifier(hidden_layer_sizes=(100), activation='relu',
                        solver='adam', max_iter=1000)
    NN1.fit(dataset1[:, :-1], dataset1[:, -1])

    # dataset2 = np.delete(dataset, [0,6,8,9,10,16,18,19,20,26,28,29], 0)

    NN2 = MLPClassifier(hidden_layer_sizes=(100), activation='relu',
                        solver='adam', max_iter=1000)
    NN2.fit(dataset2[:, :-1], dataset2[:, -1])

    NN3 = MLPClassifier(hidden_layer_sizes=(100), activation='relu',
                        solver='adam', max_iter=1000)
    NN3.fit(dataset3[:, :-1], dataset3[:, -1])

    return NN1, NN2, NN3


# -------------------------------------------------------------------------------------
# カメラ
# ID等識別して接続を安定させたいが上手くいかず。
# cap.setで画素数を変更。カメラの性能に関わらず初期値は680×400
# 外部接続(ターミナルを介して)だと初期の起動がすごく遅い、3分程度
# APIをcv2.CAP_DSHOWにすると立ち上げは早くなるが、
# デフォルトのAPIで読み取った時とアドレスの割り振りがなぜか変わる…
# Videoオブジェクトを残しながらだと4つ目のカメラ認識時になぜか抜ける
# フルHDレベルで4つカメラをセットしようとすると容量不足？になる？
# 最初の読み取り前に解像度等を設定しないと、その後変更しようとしても反映されない
# cv2.CAP_DSHOWだと画面キャプチャだけ抜ける->ハブを使うとどちらかになるため、USBポートにカメラを各1台接続する


def check_camera_connection_display(capset):
    """
    Display the image and check the camera number

    """
    true_camera_is = []
    count = 0
    # for camera_number in range(1, 6):
    for cap in capset:
        count += 1
        ret, frame = cap.read()
        if ret is True:
            start = time.time()

            while True:
                elasped_time = time.time() - start
                ret2, frame = cap.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, dsize=(400, 200))
                if elasped_time > 1.0:
                    break
                cv2.imshow(f'No. {count}', gray)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # cap.release()
            # cv2.destroyAllWindows()

            true_camera_is.append(count)
            # print("port number", count, "Find!")

        # else:
        #     print("port number", count,"None")

    print(f"Number of connected camera: {len(true_camera_is)}")

    for i in range(0, len(true_camera_is)):
        camNo_main.append(str(i+1))
        # camNo_sub.append(str(i+1))


def camsetting():
    cap = capset.pop(int(camNo_main[0])-1)
    # if camNo_main[0] > camNo_sub[0]:
    #     subcap = capset.pop(int(camNo_sub[0])-1)
    # else:
    #     subcap = capset.pop(int(camNo_sub[0])-2)
    for i in capset:
        i.release()

    ret = False
    # subret = False
    # cap = cv2.VideoCapture(int(camNo_main[0])-1)#,cv2.CAP_DSHOW)
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    time.sleep(1)
    # subcap = cv2.VideoCapture(int(camNo_sub[0])-1)#,cv2.CAP_DSHOW)
    # # # subcap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    # subcap.set(cv2.CAP_PROP_FPS, 30)
    # subcap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # subcap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # subcap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # subcap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    ret, frame = cap.read()

    # print(cap.get(cv2.CAP_PROP_FPS))
    # print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # print(subcap.get(cv2.CAP_PROP_FPS))
    # print(subcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # subret, subframe = subcap.read()
    # print(subframe.shape[0])
    # print(subframe.shape[1])

    # 接続確認
    if not ret:  # or not subret:
        root = tkinter.Tk()
        root.withdraw()
        messagebox.showerror("カメラエラー", "カメラの接続を確認してください")
        cap.release()
        # subcap.release()
        root.destroy()
        sys.exit()

    return cap  # , subcap


# -------------------------------------------------------------------------------------
# ソフトスタート
if __name__ == '__main__':
    # カメラセッティング
    print("カメラチェック中")
    print("対象のカメラを選択してください")
    capset = []
    for i in range(0, 5):
        capset.append(cv2.VideoCapture(i, cv2.CAP_DSHOW))
        # capset.append(cv2.VideoCapture(i,cv2.CAP_MSMF))
    check_camera_connection_display(capset)

    # カメラ設定GUI作成
    tki = tkinter.Tk()
    CameraSet()
    tki.mainloop()
    tki.destroy()
    cv2.destroyAllWindows()

    # GUIの設定もとにカメラ設定
    cap = camsetting()
    ret, frame = cap.read()
    # ret, subframe = subcap.read()
    # subcap.release()
    # cv2.imwrite("lib\\subcamcheck_.png", subframe)

    # SVM学習
    print("ニューラルネット学習")
    # NN1, NN2, NN3 = nn_fitting()
    # 学習済みのモデルを読み出し
    with open("lib\\NN.pkl", "rb") as f:
        NN = pickle.load(f)  # 読み出し
    with open("lib\\NN1.pkl", "rb") as f:
        NN1 = pickle.load(f)  # 読み出し
    with open("lib\\NN2.pkl", "rb") as f:
        NN2 = pickle.load(f)  # 読み出し
    with open("lib\\NN3.pkl", "rb") as f:
        NN3 = pickle.load(f)  # 読み出し

    # -------------------------------------------------------------------------------------
    # メインループ

    # GUI作成と表示の一連の流れ。mainloopでGUIからのアクションを待つ。
    tki = tkinter.Tk()  # なぜかTK()の外に書かないと効かない
    TK()
    tki.mainloop()
    tki.destroy()
    getvalue()

    # 初期設定　カメラ・マウス・変数定義
    switch_count = 0  # 照射認識用変数
    state = 0  # 試験開始フラグ
    phase = 0  # 照射開始フラグ
    diff = 0
    digit_hold = baseload
    result_count = 0
    result = "error"
    BkColor = "white"  # ポジネガ反転フラグ
    windowName = "Start/Stop[s],End[c],Rotate[r],posinega[p]"  # ウィンドウの名前
    tutorial = cv2.imread("lib\\tutorial.png")
    total_count = 0

    # トラックバー(ステータスバー)準備
    def printing(position):
        kari = 0  # おまじない
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(windowName, 640, 640)
    threshold = 5  # 2値化の閾値
    vertical = 125  # メインカメラのフォーカス
    horizontal = 125

    # トラックバー作成(右上の✕で閉じられると表示されなくなるため)
    cv2.createTrackbar("threshold", windowName, threshold, 25, printing)
    # cv2.createTrackbar("vertical", windowName, vertical, 250, printing)
    # cv2.createTrackbar("horizontal", windowName, horizontal, 250, printing)

    # コールバック関数の設定(どのウィンドウでマウス操作をするか)
    cv2.setMouseCallback(windowName, callback)

    tm = cv2.TickMeter()
    tm.start()

    count = 0
    max_count = 100
    fps = 0

    left, top, width, height = (750, 374, 257, 290)
    grab_area = {'left': left, 'top': top,
                 'width': width, 'height': height}
    sct = mss.mss()

    while True:
        # トラックバー作成(右上の✕で閉じられると表示されなくなるため)
        # cv2.createTrackbar("threshold", windowName, threshold, 25, printing)
        # cv2.createTrackbar("zoom", windowName, zoom_val, 255, printing)
        count += 1
        if count == max_count:
            tm.stop()
            fps = max_count / tm.getTimeSec()
            # print(fps)
            tm.reset()
            tm.start()
            count = 0

        # キーチェック
        key = cv2.waitKey(1) & 0xff  # キー入力を待つ
        if key != 0xff:
            # 「S」を押すと待機モードに
            if key == ord('s'):
                if state == 0:
                    # 開始時に初期化するもの
                    try:
                        os.mkdir(filepath)  # 最初のみフォルダ作成
                    except:
                        # 作成済みの場合はエラー出るが無視する
                        pass
                    header = [["frequency", "value", memo]]
                    with open(filepath+"\\temp.csv", 'w', newline="") as f:
                        writer = csv.writer(f)
                        writer.writerows(header)
                    state = 1
                    phase = 0
                    diff = 0
                    digit_hold = baseload
                elif state == 1:
                    state = 0
                    # cv2.destroyWindow("on/off")

            # 「R」を押すとトリミングエリアの回転
            if key == ord('r'):
                pt = np.vstack((pt, pt[0]))
                pt = np.delete(pt, 0, 0)

            # 「P」を押すと白黒を反転する。
            # 反転状態を基に画像を2値化
            if key == ord('p'):
                if BkColor == "white":
                    BkColor = "black"
                elif BkColor == "black":
                    BkColor = "white"

            # 照射終了、データをCSVに掃き出し、GUI復活
            if key == ord('c'):
                state = 0

                # with open(csvpath, 'w' , newline="") as f:
                #     writer = csv.writer(f)
                #     writer.writerows(data)

                if total_count > 0:
                    try:
                        os.rename(filepath+"\\temp.csv", csvpath)
                    except:
                        pass

                cv2.destroyAllWindows()
                tki = tkinter.Tk()  # なぜか外に書かないと効かない
                TK()
                tki.mainloop()
                tki.destroy()
                getvalue()  # 測定継続なら条件読み込み
                cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(windowName, 640, 640)
                cv2.setMouseCallback(windowName, callback)
                cv2.createTrackbar("threshold", windowName,
                                   threshold, 25, printing)
                # cv2.createTrackbar("vertical", windowName,
                #                    vertical, 250, printing)
                # cv2.createTrackbar("horizontal", windowName,
                #                    horizontal, 250, printing)
                total_count = 0

        # トラックバー作成(右上の✕で閉じられると表示されなくなるため)
        try:
            cv2.getWindowProperty(windowName, cv2.WND_PROP_AUTOSIZE)
        except:
            cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(windowName, 640, 640)
            cv2.setMouseCallback(windowName, callback)
            cv2.createTrackbar("threshold", windowName,
                               threshold, 25, printing)
            # cv2.createTrackbar("vertical", windowName, vertical, 250, printing)
            # cv2.createTrackbar("horizontal", windowName,
            #                    horizontal, 250, printing)
            pass

        # トラックバーの値を読み取る
        # 作成直後だと読み取れないからキーチェックの後に入れる
        if cv2.getTrackbarPos("threshold", windowName) >= 0:  # ウィンドウを✕で消すと値がバグる対策
            threshold = cv2.getTrackbarPos("threshold", windowName)
            # vertical = cv2.getTrackbarPos("vertical", windowName)
            # horizontal = cv2.getTrackbarPos("horizontal", windowName)

        # カメラ・PCキャプチャ画面の読み込み
        ret, frame = cap.read()
        # frame = frame[395-vertical:935-vertical, 730 -
        #               horizontal*2:1690-horizontal*2]  # 疑似的にズーム(エリア操作簡易化のため)
        frame = cv2.resize(frame, dsize=(xmax, ymax))
        capture = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # グレースケールに変換

        # 指定したエリアを長方形に切り抜く
        perspective1 = np.float32([pt[0], pt[1], pt[2], pt[3]])  # 指定した4隅の座標
        perspective2 = np.float32(
            [[0, 0], [210, 0], [210, 110], [0, 110]])  # 変換後のサイズ
        psp_matrix = cv2.getPerspectiveTransform(
            perspective1, perspective2)  # 変換用行列作成
        try:
            img_psp = cv2.warpPerspective(
                capture, psp_matrix, (210, 110))  # 行列を元にホモグラフィ変換
            img_psp = img_psp[5:105, 5:205]  # 引き伸ばした際の枠周りのギザギザを削除
        except:
            # 何故かエラーが出ることがあるが無視できる
            continue

        # トリミングエリアを画面上に表示
        cv2.line(frame, (pt[0, 0], pt[0, 1]),
                 (pt[1, 0], pt[1, 1]), (0, 0, 255), 5)
        cv2.line(frame, (pt[1, 0], pt[1, 1]),
                 (pt[2, 0], pt[2, 1]), (0, 255, 255), 5)
        cv2.line(frame, (pt[2, 0], pt[2, 1]),
                 (pt[3, 0], pt[3, 1]), (0, 255, 255), 5)
        cv2.line(frame, (pt[3, 0], pt[3, 1]),
                 (pt[0, 0], pt[0, 1]), (0, 255, 255), 5)

        # フォーカス値をメインカメラに反映
        # cap.set(cv2.CAP_PROP_FOCUS, focus_val)

        # 前処理
        img_prepro, median = preprocessing(img_psp, seg, threshold)

        # 数字エリアを決定
        digits_positions = find_digits_positions(img_prepro)

        # エリアから数字を判断し、結果を画像に表示
        num = recognize_digits(digits_positions, median, seg)

        # 計算
        digits = cul_digits(num)

        # 結果表示用の画像用意
        output = disp_process(img_psp, digits_positions, digits, num, result)

        # subframe = subcap.read()[1]
        if count % 5 == 1:
            # subframe = cv2.cvtColor(
            #     subcap.read()[1], cv2.COLOR_BGR2GRAY)  # グレースケールに変換
            # スクリーンショット取得
            # with mss.mss() as sct:

            subframe = cv2.cvtColor(
                np.array(sct.grab(grab_area)), cv2.COLOR_BGRA2GRAY)
            # img = np.array(img)[:, :, :3]
            # subframe = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 照射開始「S」押下後
            # TimingBox = cv2.resize(
            #     subframe[643:700, 1384:1453], (200, 100))  # 東京1920
            # fBox_disp = cv2.resize(
            #     subframe[395:420, 1123:1280], (200, 100))  # 東京1920
            # TimingBox = cv2.resize(
            #     subframe[420:460, 920:990], (200, 100))  # 東京1280
            # fBox_disp = cv2.resize(
            #     subframe[255:290, 750:840], (200, 100))  # 東京1280
            TimingBox = cv2.resize(
                subframe[height-60:height, width-70:width], (120, 100))  # 東京1280
            fBox_disp = cv2.resize(
                subframe[0:30, 0:100], (120, 100))  # 東京1280

        if state == 1:
            # PCキャプチャから(AM変調中)が表示されているかを検知
            # 位置を指定しているのでウィンドウを動かさないで使用(改善余地)
            # RGB平均値を出力
            # flattenで一次元化し緑と赤の画素値平均を取得
            # g = TimingBox.T[1].flatten().mean()
            # r = TimingBox.T[2].flatten().mean()
            g = TimingBox.flatten().mean()
            # 緑画素と赤画素に差がある状態を照射中と判定(ノイズ対策で3フレーム続いたらON/OFF切り替えとしている)
            # if g-r > 50:
            if g < 200:  # ------------------------------------------------------------------------------------検討
                switch_count = switch_count+1
                switch_count = min(3, switch_count)
            else:
                switch_count = switch_count-1
                switch_count = max(0, switch_count)
            if switch_count == 3 and phase == 0:
                phase = 1
                # 位置を指定しているのでウィンドウを動かさないで使用(改善余地) ウィンドウを検知する等
                # エリアをグレースケール化→2値化後にテンプレートマッチングへ
                # fBox = cv2.cvtColor(
                #     subframe[265:278, 750:840], cv2.COLOR_BGR2GRAY)
                # cv2.imwrite("lib\\fbox_.png", fBox)
                # fBox = cv2.cvtColor(
                #     fBox_disp, cv2.COLOR_BGR2GRAY)
                ret, fBox_disp = cv2.threshold(
                    fBox_disp, 190, 255, cv2.THRESH_BINARY)
                # cv2.imwrite("lib\\fbox.png", fBox)
                freq = freq_check(fBox_disp)
            elif switch_count == 0 and phase == 1:
                phase = 2

            # 照射中に初期負荷から1番離れた値を保持
            # ノイズ対策、画面表示安定待ちのため8フレーム続いたら正規の値として認める
            if phase == 1:
                if digit_hold == digits:
                    result_count += 1
                    if result_count > 5:
                        result_count = 0
                        try:
                            if diff <= abs(digits-baseload):
                                diff = abs(digits-baseload)
                                result = digits
                        except:
                            result = digits
                else:
                    result_count = 0

                digit_hold = digits

            # OFFになったら保持した値をリストに追記していく
            elif phase == 2:
                try:
                    with open(filepath+"\\temp.csv", 'a', newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([freq, result])
                        total_count += 1
                except:
                    pass
                print([freq, result])
                phase = 0
                diff = 0
                digit_hold = baseload
                result = "error"

        # 処理過程を結合して表示
        # 配列次元を合わせるために、BGRへ変換後結合
        frame = cv2.vconcat([frame, cv2.hconcat([output, cv2.cvtColor(img_prepro, cv2.COLOR_GRAY2BGR),
                            cv2.cvtColor(TimingBox, cv2.COLOR_GRAY2BGR), cv2.cvtColor(fBox_disp, cv2.COLOR_GRAY2BGR)])])  # , tutorial
        # FPS表示(トリミングエリアの入力後).
        cv2.putText(frame, str(int(fps)), (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 128, 128), 2)
        cv2.imshow(windowName, frame)

    # キャプチャをリリースして、ウィンドウをすべて閉じる

    cap.release()
    # subcap.release()
    cv2.destroyAllWindows()


# 以下遺産-----------------------------------------------


# Videoオブジェクトを一通り作ってから選択する。
# CAP_DSHOWだとオブジェクトの数に制限がある？
# def camera_set(capset, mcam, scam):
#     global cap, subcap, ret, subret
#     cap = capset.pop(int(mcam)-1)
#     if camNo_main[0] > camNo_sub[0]:
#         subcap = capset.pop(int(scam)-1)
#     else:
#         subcap = capset.pop(int(scam)-2)
#     for i in capset:
#         i.release()

# 位置推定


# def serch_frq(frame):
#     temp = cv2.imread("lib\\subtarget.jpg", cv2.IMREAD_GRAYSCALE)
#     temp1 = cv2.imread("lib\\subtarget1.jpg", cv2.IMREAD_GRAYSCALE)

#     # 周波数のエリアを持ってきてテンプレートマッチング　X座標の小さい順に並べて返す
#     res1 = cv2.matchTemplate(frame, temp, cv2.TM_CCOEFF_NORMED)
#     x1, y1 = np.where(res1 >= 0.9)
#     minVal, maxVal1, minLoc, maxLoc1 = cv2.minMaxLoc(res1)
#     res2 = cv2.matchTemplate(frame, temp1, cv2.TM_CCOEFF_NORMED)
#     y2, x2 = np.where(res2 >= 0.9)
#     minVal, maxVal2, minLoc, maxLoc2 = cv2.minMaxLoc(res2)

    # if maxVal1 > maxVal2:
    #     xs, ys = maxLoc1
    #     xl =
    #     yl =
    # else:
    #     xs, ys = maxLoc2
    #     xl =
    #     yl =
    # return xs, ys
