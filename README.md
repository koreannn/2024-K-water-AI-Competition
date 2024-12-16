# DNN, ML 기반 이상탐지 알고리즘 개발

- Target : 누수가 발생한 위치를 특정할 수 있는, 직접적으로 관련이 있는 압력계
  - 관망의 상위 지점에서 이상이 발생할 경우, 하류에도 영향이 가는것이 맞지만, 직접적인 지점을 특정하는 것이 목표이므로, 하류는 고려하지 않아도 됨
  - 즉, "이상 감지 + 누수 발생 위치를 정확히 특정하는 것" 이 목표
      
\[Experience]
- (Train)Input : Train 데이터셋 수도관망에 있는 각 압력계 지점에 대한 분단위 압력값
- (Inference)Input : Test 데이터셋 수도관망에 있는 각 압력계 지점에 대한 분단위 압력값
- Output : 수도관망에 있는 압력계 지점 중 이상발생 시점 및 지점 탐지 $\to$ 각 지점에 대한 이상치(Anomaly)값 0, 1로 분류
    
\[Task, Performance]
- Anomaly 발생 시점과 지점을 최대한 정확하게 감지해내야 함.

### 데이터 명세
![image](https://github.com/user-attachments/assets/3bda4f89-0f9a-484b-9173-30c229d63c53)

# DNN 기반 이상탐지



# ML기반 이상탐지
