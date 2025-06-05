# FreqAI strategies

# 安装依赖
```shell
pip install optuna stable-baselines3 sb3-contrib
pip install optuna-integration
```

# 下载回测数据
```shell
freqtrade download-data --config user_data/config-cpu.json --timerange 20240101-20250602 --timeframes 5m 15m 1h 4h 1d
```

```shell
freqtrade backtesting --config user_data/config-cpu.json --strategy QuickAdapterV3 --freqaimodel QuickAdapterRegressorV3 --timerange 20250403-20250603
```
# 开启交易 gpu
```shell
freqtrade trade --config user_data/config-gpu.json --strategy QuickAdapterV3 --freqaimodel QuickAdapterRegressorV3 --logfile user_data/logs/freqtrade.log
```


