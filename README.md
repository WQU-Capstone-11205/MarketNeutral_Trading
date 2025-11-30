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
