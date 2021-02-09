# Gradient Descent
In this repository, I have implemented gradient descent algorithm from scratch in python.


![image_0](https://github.com/pradyyadav/Images/blob/main/3.gif?raw=True)
![image_4](https://cdn.mathpix.com/snip/images/Qej8y_MlWd1wo2yF2f72g6_LzZnMd0BUAtkxBqYJsUo.original.fullsize.png)
![image_3](https://cdn.mathpix.com/snip/images/9R8XqShEOMfFa-iYbNyvUBKbEx92-RDYIciDDBAhURk.original.fullsize.png)
![image_2](https://cdn.mathpix.com/snip/images/8w1pnRPBj1cOOkDxL8WljMVOoO3MKC7R6oOb9ecyoG0.original.fullsize.png)
![image_1](https://cdn.mathpix.com/snip/images/lADpZD7IZoGybL8dQD9sa2OraBy-9zzjEbtJ_2Cn-W0.original.fullsize.png)

##### This is the python implementation of the algoriithm


```
def gradient_descent(X,y):
    m = b = 0
    learning_rate = 0.01      # Small steps by which the cost function is optimised along the slope
    epochs = 1000             # The iterations during which cost function is optimised
    n = len(X)
    for epoch in range(epochs):
        y_predicted = m*X + b                           # Predicted function with respect to features 
        cost = (1/(2*n))*sum([val**2 for val in (y_predicted-y)])  # Cost function
        derivative_wrt_m = (1/n)*sum((y_predicted-y)*X) # Gives the slope of cost function along m
        derivative_wrt_b = (1/n)*sum(y_predicted-y)     # Gives the slope of cost function along b 
        m = m - learning_rate * derivative_wrt_m        # Steps by which cost is minimized along m
        b = b - learning_rate * derivative_wrt_b        # Steps by which cost is minimized along b 
        print("m: {},  b: {},  Epoch: {},  cost: {}   ".format(m,b,epoch,cost))
    return m,b,cost
```
##### You can find my medium blog [here](https://pradyyadav.medium.com/gradient-descent-c37ef7011a2f). I have explained each and every concept in a beginner friendly manner.
