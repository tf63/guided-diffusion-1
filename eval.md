### 定量評価

**t = 500**
| Class | Method | MSE $(\times 10^{-3})$ | Top-1 Accuracy (\%) | Top-5 Accuracy (\%) |
| ------------ | -------- | ---------------------- | ------------------- | ------------------- |
| airplane2car | Baseline | 8.2 | 2.9 | 11.8 |
| airplane2car | Classifier Guidance | 8.9 | 8.4 | 26.0 |
| airplane2car | Ours | 12.5 | 28.3 | 56.8 |
| car2airplane | Baseline | 13.5 | 17.9 | 34.8 |
| car2airplane | Classifier Guidance | 14.1 | 34.5 | 55.8 |
| car2airplane | Ours | 18.8 | 77.4 | 92.2 |

---

**t = 400**
| Class | Method | MSE $(\times 10^{-3})$ | Top-1 Accuracy (\%) | Top-5 Accuracy (\%) |
| ------------ | -------- | ---------------------- | ------------------- | ------------------- |
| airplane2car | Baseline | 5.7 | 0.9 | 5.8 |
| airplane2car | Classifier Guidance | 6.2 | 3.6 | 14.1 |
| airplane2car | Ours | 8.4 | 16.3 | 39.5 |
| car2airplane | Baseline | 9.5 | 7.9 | 19.8 |
| car2airplane | Classifier Guidance | 9.9 | 18.7 | 34.9 |
| car2airplane | Ours | 13.1 | 57.7 | 78.3 |

---

**t = 300**
| Class | Method | MSE $(\times 10^{-3})$ | Top-1 Accuracy (\%) | Top-5 Accuracy (\%) |
| ------------ | -------- | ---------------------- | ------------------- | ------------------- |
| airplane2car | Baseline | 3.8 | 0.3 | 2.8 |
| airplane2car | Classifier Guidance | 4.0 | 1.2 | 6.6 |
| airplane2car | Ours | 5.3 | 6.3 | 21.5 |
| car2airplane | Baseline | 6.3 | 3.6 | 10.4 |
| car2airplane | Classifier Guidance | 6.5 | 7.5 | 17.3 |
| car2airplane | Ours | 8.4 | 31.5 | 50.9 |

---
**(time)**
| Class | Method | MSE $(\times 10^{-3})$ | Top-1 Accuracy (\%) | Top-5 Accuracy (\%) |
| ------------ | -------- | ---------------------- | ------------------- | ------------------- |
| airplane2car | Baseline ($t_s = 500$) | 8.2 | 2.9 | 11.8 |
| airplane2car | Classifier Guidance ($t_s = 500$) | 8.9 | 8.4 | 26.0 |
| airplane2car | Ours ($t_s = 300$) | 5.3 | 6.3 | 21.5 |
| airplane2car | Ours ($t_s = 400$) | 8.4 | 16.3 | 39.5 |
| airplane2car | Ours ($t_s = 500$) | 12.5 | 28.3 | 56.8 |


| Class | Method | MSE $(\times 10^{-3})$ | Top-1 Accuracy (\%) | Top-5 Accuracy (\%) |
| ------------ | -------- | ---------------------- | ------------------- | ------------------- |
| car2airplane | Baseline ($t_s = 500$) | 13.5 | 17.9 | 34.8 |
| car2airplane | Classifier Guidance ($t_s = 500$) | 14.1 | 34.5 | 55.8 |
| car2airplane | Ours ($t_s = 300$) | 8.4 | 31.5 | 50.9 |
| car2airplane | Ours ($t_s = 400$) | 13.1 | 57.7 | 78.3 |
| car2airplane | Ours ($t_s = 500$) | 18.8 | 77.4 | 92.2 |

---
**Ours (time)**
| Class | Method | MSE $(\times 10^{-3})$ | Top-1 Accuracy (\%) | Top-5 Accuracy (\%) |
| ------------ | -------- | ---------------------- | ------------------- | ------------------- |
| airplane2car | Ours ($t_s = 300$) | 5.3 | 6.3 | 21.5 |
| airplane2car | Ours ($t_s = 400$) | 8.4 | 16.3 | 39.5 |
| airplane2car | Ours ($t_s = 500$) | 12.5 | 28.3 | 56.8 |
| car2airplane | Ours ($t_s = 500$) | 8.4 | 31.5 | 50.9 |
| car2airplane | Ours ($t_s = 500$) | 13.1 | 57.7 | 78.3 |
| car2airplane | Ours ($t_s = 500$) | 18.8 | 77.4 | 92.2 |

---
