<h1>Painting-Diary</h1>
<hr>
	<h3>임의의 사진을 입력받아 특정 그림체의 그림으로 바꿔주는 프로젝트입니다.</h3>
<pre>목차 
1. 프로젝트 개요
2. 참고한 논문 및  github code, dataset
3. 프로젝트 변경사항
4. 개선점
</pre>
<h3>프로젝트 개요</h3>
<pre>
입력이미지를 받으면 모델에서 사용하는 그림체로 이미지의 style을 바꿔주는 프로젝트입니다.
해당 프로젝트 모델에서 사용하는 그림체는 '톰과 제리' 그림체입니다. 
예를 들면 아래와 같이 ---을 받아 톰과제리 그림작가가 ---을 그린것과 같은 결과를 내놓는 모델입니다.
참고한 모델과 깃헙 코드가 있지만 해당 프로젝트의 목적과는 많은 차이가 있어 많은부분을 수정하였습니다.
</pre>
<h3>참고 논문(style transfer 기본 컨셉트) 및 github code</h3>
<pre>
참고 논문 : https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf
참고 깃헙 : https://github.com/fawazsammani/The-Complete-Neural-Networks-Bootcamp-Theory-Applications/blob/master/Neural%20Style%20Transfer.ipynb
</pre>
<pre>
먼저 Neural style transfer의 기본적인 컨셉은 그리고 싶은 객체인 content image와 적용하고싶은 그림체를 담은 style image 두개를 input으로 넣습니다.
그리고 나서 원하는 output을 그릴 canvas 인 새로운 이미지 x를 초기화합니다. (보통은 gaussian noise로 초기화합니다.)
전체적인 모델의 inference 과정은, pretrain된 VGG16 모델을 불러옵니다. 이를 feature extractor로 사용하는데요, 
먼저 content image를 VGG에 통과시키고 VGG 의 뒷부분 layer의 출력값을 저장합니다. 이를 VGG(content)라고 하겠습니다.
그리고나서 이번에는 style image를 똑같은 VGG에 넣어 계산을하는데요, style image를 계산할때에는 중간중간에 있는 몇 개의 layer에 대한 값을 저장합니다.
이를 VGG(style) 이라고 하겠습니다. 이 다음에 VGG에 x를 넣어 계산을 하는데, 이를 VGG(x)라고  하겠습니다. 
이렇게 세 종류의 값을 구하면, loss를 구하는데요, 먼저 content loss로 VGG(content) 와 VGG(x) 사이의 거리를 content loss로 합니다.
다음으로 style loss를 구하기 위해서, 본 논문의 저자는 feature 값 들의 '관계'가 style을 특정한다고 했습니다. 그렇기때문에 단순 VGG의 출력값이 아니라 VGG출력값의 gram matrix를 구하여 gram(VGG(style)) 과 gram(VGG(x)) 의 거리를 style loss로 사용합니다.
최종적인 loss 는 
loss = alpha * content_loss + beta * style_loss 형태로 되며, gradient descent를 이용하여 parameter를 갱신하는 것은 오직 x의 값만을 갱신합니다. 이 과정을 지정한 epoch 만큼 진행을 하면 loss가 점점 줄게되면서 content와 style의 loss 비율에 맞게 x가 변화되며 이 optimization 결과가 x에 나오게됩니다.
</pre>