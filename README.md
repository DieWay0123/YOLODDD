# 基於YOLOv11和MediaPipe的疲勞駕駛檢測

## 流程
使用MediaPipe偵測左右眼和嘴巴的位置 -> Crop出左右眼和嘴巴 -> 將左右眼和嘴巴分別餵給偵測兩者開合的模型偵測 -> 透過演算法進行疲勞檢測

## 備忘錄
1. 封裝擷取左右眼和嘴巴的函數
2. 測試對資料集用cv技術進行前處理前後的accuracy和loss