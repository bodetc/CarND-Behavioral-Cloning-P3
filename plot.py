import matplotlib.pyplot as plt
import numpy as np

x=np.arange(1,11)
test      =np.array([279.,214.,200.,186.,177.,169.,162.,158.,153.,150.])
validation=np.array([214.,194.,190.,176.,177.,173.,172.,172.,163.,170.])

# Save histogram
plt.figure(figsize=(10, 6))
plt.plot(x,0.0001*test, label='Test loss')
plt.plot(x,0.0001*validation, label='Validation loss')
plt.xlabel('Traffic sign')
plt.ylabel('Loss')
plt.legend()
plt.savefig('writeup/loss.png')