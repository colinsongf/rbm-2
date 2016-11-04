# MNIST

## Guidelines

- large filter size works better (12>5>3)
- filters need to overlap
- but 1\*1 would bring too many parameters, so 2\*2 seems to be a good choice
- learning rate?
- pcd k>5
- sometimes may have dull outputs (all 0, all 1...), but more epochs generally yield better results
- a relatively larger batch size would give better results (20 > 10)



## Interesting facts that we ~~wont' understand ever~~ don't understand yet

- half of the filters visualized are identical and useless
- legit degit would be shaken away after some iterations, so when would it converge (or never)?  Converge means eigenvector?
- ​

## HuNet

> (['conv',(12,12,1,64), (2,2)]
> ['conv',(5,5,64,128),(2,2)],
> ['fc', 500])

#### Evolution plot

row: vis per 100 iterations with raw input at first row

- 50

 ![hu_epoch49-random-evo](evo/hu_epoch49-random-evo.png) ![hu_epoch49-test](evo/hu_epoch49-test.png)

- 55/60

not so good, many 0 and 1

#### Filter Visualizations

at epoch 50

- `w0: (12,12,1,64)` a lot of them seem to be useless?

 ![hu_epoch49_w0](evo/hu_epoch49_w0.png)

- w1: (5,5,64,64)`  too small... and second layers filter usually don't make lots of sense

 ![hu_epoch49_w1](evo/hu_epoch49_w1.png)



## Hu-32

#### Evolution Plot

- 45

 ![32_epoch44-random-evo](evo/32_epoch44-random-evo.png)

 ![32_epoch44-test](evo/32_epoch44-test.png)



#### Filter

- `w0: (12,12,1,32)`  the redundant filters look fucking same as HuNet's?

 ![hu_epoch49_w0](evo/dbm_mini_epoch44_w0.png)

Deeper layers make no sense.

## Hu-128

#### Evolution Plot

- 55

 ![128_epoch55-random-evo](evo/128_epoch55-random-evo.png)

![128_epoch55-test](evo/128_epoch55-test.png)

- 50/60 

not so good, many 0 and 1



#### FC

forgot the actual configuration, but they should look similar...

- epoch 40 with pcd k=10

 ![epoch38-plot](/home/eric/Develop/10715/rbm/fc10_dbm_0/epoch38-plot.png)

- epoch 10 with pcd k=100

 ![pcd_100_epoch10-plot](fc/pcd_100_epoch10-plot.png)



## Some other random results

obtained at epoch 20, using pcd with k=10

- default

```
(['conv',(5,5,1,64), (2,2)], #0
 ['conv',(5,5,64,64),(2,2)],
 ['fc', 512])
```

 ![mnist_0_epoch19-plot](mnist/mnist_0_epoch19-plot.png)

- increase stride

```
(['conv',(5,5,1,64), (3,3)], #1
 ['conv',(5,5,64,64),(3,3)],
 ['fc', 512])
```

![mnist_0_epoch19-plot](mnist/mnist_1_epoch19-plot.png)

- and more 

```
(['conv',(5,5,1,64), (4,4)], #2
 ['conv',(5,5,64,64),(4,4)],
 ['fc', 512])
```

![mnist_0_epoch19-plot](mnist/mnist_2_epoch19-plot.png)

- stride = size

```
(['conv',(5,5,1,64), (5,5)], #3
 ['conv',(5,5,64,64),(5,5)],
 ['fc', 512])
```

![mnist_0_epoch19-plot](mnist/mnist_5_epoch19-plot.png)

- more fc units

```
(['conv',(5,5,1,64), (2,2)], #4
 ['conv',(5,5,64,64),(2,2)],
 ['fc', 1024])
```

![mnist_0_epoch19-plot](mnist/mnist_4_epoch19-plot.png)

- more filters

```
(['conv',(5,5,1,64), (2,2)], #5
 ['conv',(5,5,64,128),(2,2)],
 ['fc', 512])
```

![mnist_0_epoch19-plot](mnist/mnist_5_epoch19-plot.png)

- and more filters

```
(['conv',(5,5,1,128), (2,2)], #6
 ['conv',(5,5,128,128),(2,2)],
 ['fc', 512])
```

all black

- stride  = 1

```
(['conv',(3,3,1,64), (1,1)], #7
 ['conv',(3,3,64,128),(1,1)],
 ['fc', 512])
```

explode, all white output

- smaller filter size

worse than 5*5

```
(['conv',(3,3,1,64), (2,2)], #8
 ['conv',(3,3,64,64),(2,2)],
 ['fc', 512])
```

![mnist_0_epoch19-plot](mnist/mnist_8_epoch19-plot.png)

- stride =  filter size again

```
(['conv',(3,3,1,64), (3,3)], #9
 ['conv',(3,3,64,64),(3,3)],
 ['fc', 512])
```

![mnist_0_epoch19-plot](mnist/mnist_9_epoch19-plot.png)

- deeper

might need more epoch

```
(['conv',(5,5,1,32), (2,2)], #10
 ['conv',(5,5,32,32),(2,2)],
 ['conv',(3,3,32,64),(2,2)],
 ['conv',(3,3,64,128),(2,2)],
 ['fc',512])
```

![mnist_0_epoch19-plot](mnist/mnist_10_epoch19-plot.png)

- upside down for fun

40 epochs

```
(['conv',(3,3,1,32), (2,2)], #11
 ['conv',(3,3,32,32),(2,2)],
 ['conv',(5,5,32,64),(2,2)],
 ['fc',1024])
```

![mnist_0_epoch19-plot](mnist/mnist_11_epoch39-plot.png)

- and more filters

```
(['conv',(5,5,1,64), (2,2)], #12
 ['conv',(5,5,64,128),(2,2)],
 ['conv',(5,5,128,128),(2,2)],
 ['fc',1024])
```

all white

- different filter size

40 epochs

```
(['conv',(4,4,1,64), (3,3)], #13
 ['conv',(4,4,64,128),(3,3)],
 ['conv',(4,4,128,128),(3,3)],
 ['fc',1024])
```

![mnist_0_epoch19-plot](mnist/mnist_13_epoch39-plot.png)















