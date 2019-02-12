## 各種金融指標的實作會放在這邊

簡易的使用方式如下：
1. 建立捷徑在自己的目錄下, 假設你的目標目錄是larry/

    `$ ln -s $PWD/etf_tools larry`


2. 程式碼範例：

    ```python
    from etf_tools import kd_rsv
    erf_0050 = tetfp[tetfp['code'] == '0050']
    feat_kd_rsv = kd_rsv(erf_0050.close,
                         high=erf_0050.high,
                         low=erf_0050.low,
                         n_days=9,
                         k_init=0.5, k_w=0.33,
                         d_init=0.5, d_w=0.33)
    feat_kd_rsv.head()
    ```

