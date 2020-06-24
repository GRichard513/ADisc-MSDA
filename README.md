# ADisc-MSDA
Code for Unsupervised Multi-Source Domain Adaptation for Regression [1] paper

---

Every experiment was made using CUDA Drivers 9.0 and a Linux machine.
Non GPU users can turn it down by specifying the device to 'cpu'.
To reproduce our environment, one can use  the following line: **conda env create -f environment.yml**

Data for the Amazon experiments can be downloaded and processed by running create_amazon.py
Code also includes **DANN** [2] and **MDAN** [3]. For [3], the code was largely inspired from their original implementation (https://github.com/KeiraZhao/MDAN).

---

[1] Richard, G., de Mathelin, A., Hébrail, G., Mougeot, M., & Vayatis, N. (2020). Unsupervised Multi-Source Domain Adaptation for Regression.

[2] Ganin, Y., Ustinova, E., Ajakan, H., Germain, P., Larochelle, H., Laviolette, F., ... & Lempitsky, V. (2016). Domain-adversarial training of neural networks. The Journal of Machine Learning Research, 17(1), 2096-2030.

[3] Zhao, H., Zhang, S., Wu, G., Moura, J. M., Costeira, J. P., & Gordon, G. J. (2018). Adversarial multiple source domain adaptation. In Advances in neural information processing systems (pp. 8559-8570).
