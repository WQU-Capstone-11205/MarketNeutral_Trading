<hl>**MarketNeutral_Trading: Long/Short pair trading strategy using Hybrid ML model**</hl>

Created date: Sept 10, 2025

Created by: Santosh More

<hl>**Introduction:**</hl>

This Hybrid model is made of BOCPD + VAE + RL models</p>

<p>The BOCPD algorithm is based on the following paper:
  
  Adams, Ryan Prescott, and David JC MacKay. "Bayesian online changepoint detection." arXiv preprint arXiv:0710.3742 (2007).</p>

The VAE is referred from <a href="https://www.ibm.com/think/topics/variational-autoencoder">here<a>

The Reinforced Learning model is basically an Actor-Critic model.

<hl>**Project folder structure:**</hl>
<p><a href="https://github.com/WQU-Capstone-11205/MarketNeutral_Trading/tree/wqu_dev_branch/Notebooks">Notebooks:</a> All Hybrid and traditional model's Jupyter notebooks are located here </p>
<p><a href="https://github.com/WQU-Capstone-11205/MarketNeutral_Trading/tree/wqu_dev_branch/backtest">backtest:</a> Evaluation loops for all Hybrid models are located here </p>
<p><a href="https://github.com/WQU-Capstone-11205/MarketNeutral_Trading/tree/wqu_dev_branch/data_loading">data_loading:</a> Data loading modules from external sources are located here</p>
<p><a href="https://github.com/WQU-Capstone-11205/MarketNeutral_Trading/tree/wqu_dev_branch/metrics">metrics:</a> All Hybrid and traditional model's metrics modules are located here</p>
<p><a href="https://github.com/WQU-Capstone-11205/MarketNeutral_Trading/tree/wqu_dev_branch/ml_dl_models">ml_dl_models:</a> All base models of VAE, CNN-LSTM, RL, Transformer are located here</p>
<p><a href="https://github.com/WQU-Capstone-11205/MarketNeutral_Trading/tree/wqu_dev_branch/plots">plots:</a> All Hybrid and traditional model plot modules are located here</p>
<p><a href="https://github.com/WQU-Capstone-11205/MarketNeutral_Trading/tree/wqu_dev_branch/structural_break">structural_break:</a> Structural break detection modules are located here</p>
<p><a href="https://github.com/WQU-Capstone-11205/MarketNeutral_Trading/tree/wqu_dev_branch/trad_arbt_strat">trad_arbt_strat:</a> Taditional Arbitrage modules are located here </p>
<p><a href="https://github.com/WQU-Capstone-11205/MarketNeutral_Trading/tree/wqu_dev_branch/train">train:</a> Training loop for Hybrid models are located here</p>
<p><a href="https://github.com/WQU-Capstone-11205/MarketNeutral_Trading/tree/wqu_dev_branch/tuning">tuning:</a> Tuning loop for Hybrid models are located here</p>
<p><a href="https://github.com/WQU-Capstone-11205/MarketNeutral_Trading/tree/wqu_dev_branch/util">util:</a> All utility modules like replay buffer, RMS, model IO, benchmark returns, seed random, etc. are located here</p>

<hl>**Installation:**</hl>
    <p>pip install -r requirements.txt</p>

<hl>**Instructions (from Colab Notebook):**</hl> 
    <p>!git clone https://github.com/WQU-Capstone-11205/MarketNeutral_Trading.git</p>
    <p>%cd /content/MarketNeutral_Trading</p>

<hl>**Notebooks Rundown:**</hl>
<p>1) Enhanced_Traditional_econometrics_methods.ipynb: In this Notebook, Step 4 has metrics for Cointegration spread and Step 6 for z-score spread.</p>
<p>2) Enhanced_BOCPD_VAE_RL_pipeline_tune_train_test.ipynb, Enhanced_BOCPD_VAE_CNN_LSTM_pipeline_tune_train_test.ipynb & Enhanced_BOCPD_VAE_TRAFO_pipeline_tune_train_test.ipynb: In these Notebooks, Step 6 and Step 8 have relevant metrics. Step 7 has BOCPD, VAE and Profit and Loss plots.</p>
<p>4) The Run All command of Colab Notebook will run all cells and will display similar results. The results might differ because of the Stochastic nature of models, the tensors used, even though we have tried to limit the randomness at appropriate places, but the results may differ. It’s good to try a few times to watch the results.</p>
<p>5) The Hybrid model Colab Notebook files contain all required features of the Hybrid model:</p>
  <p>Step1: Github setup</p>
  <p>Step2: Data loading</p>
  <p>Step3: Hyperparameters tuning</p>
  <p>Step4: Training Hybrid model</p>
  <p>Step5: Testing model</p>
  <p>Step6: Display Test metrics</p>
  <p>Step7: Display plots</p>
  <p>Step8: Stabalization and Adaptability metrics</p>

