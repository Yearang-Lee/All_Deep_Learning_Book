# Convolutional Neural Network ( CNN )

> 입력된 이미지에서 다시 한번 특징을 추출하기 위해 filter를 도입하는 기법

<img src="/image/4.png" style="zoom:60%;" />

이미지 : <img src="/image/5.png" style="zoom:60%;" />

```python
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(28,28,1), activation = 'relu'))
```

- ```Conv2D``` : 컨볼루션 층을 추가하는 함수
  - 32 : filter를 몇 개 적용할지
  - `kernel_size` : filter의 크기
  - `input_shape` : 입력되는 값 ( 행, 열, 색상 또는 흑백(색상:3, 흑백:1) ) 
  - `activation` : 활성화 함수

```python
 model.add(Con2D(64, (3,3), activation = 'relu'))
```

- filter 64개를 적용한 컨볼루션 층 추가

```python
model.add(MaxPooling2D(pool_size = 2 ))
```

(앞의 결과가 여전히 크고 복잡하기 때문에 다시 한 번 축소해야 한다. => pooling)

- `pool_size` : pooling 창의 크기 ( 2로 정하면 전체 크기가 절반으로 줄어든다. )

```python
model.add(Dropout(0.25))
```

(랜덤하게 노드를 제외시키면 과적합을 방지 할 수 있다.)

- `Dropout(0.25)` : 25%의 노드를 제외시킨다.

```python
model.add(Flatten())
```

- 컨볼루션 층이나 맥스 풀링은 이미지를 2차원 배열인 채로 다룬다. => 이를 1차원으로 바꿔주는 함수 : ```Flatten()```



#### 컨볼루션 신경망 전체 코드

```python
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(28,28,1), activation = 'relu'))
model.add(Con2D(64, (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2 ))
model.add(Dropout(0.25))
model.add(Flatten())
# 앞에서 Dense() 함수를 이용해 만들었던 기본 층에 연결
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation = 'softmax'))
          

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

<img src="/image/6.png" style="zoom:60%;" />



