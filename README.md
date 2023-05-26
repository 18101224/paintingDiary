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
임의의 style image를 받아 style transfer를 적용하는 기존의 모델을 수정하여
한가지 style 을 적용하게끔하고, style에 더 집중한 모델을 만들었습니다.
해당 프로젝트 모델에서 사용하는 그림체는 미국 애니메이션 그림체입니다.
참고한 모델과 깃헙 코드가 있지만 해당 프로젝트의 목적과는 많은 차이가 있어 많은부분을 수정하였습니다.
</pre>
<h3>참고 논문(style transfer 기본 컨셉트) 및 github code</h3>
<pre>
참고 논문 : https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf
참고 깃헙 : https://github.com/fawazsammani/The-Complete-Neural-Networks-Bootcamp-Theory-Applications/blob/master/Neural%20Style%20Transfer.ipynb
</pre>

먼저 Neural style transfer의 기본적인 컨셉은 그리고 싶은 객체인 content image와 적용하고싶은 그림체를 담은 style image 두개를 input으로 넣고,<br>
각 이미지를 pretrained VGG 에 넣은 feature 값과 결과가 될 이미지를 VGG에 넣은 값 사이의 mse를 각각의 loss로 사용합니다.
이러한 style 이미지에 대한 style loss와 content이미지에 대한 content loss를 weighted sum을 한 것을 loss로 사용해 결과이미지의  픽셀값을 optimization합니다.  
결과이미지는 content image를 clone한 것으로부터 시작해서 loss를 줄여나가는 방향으로 optimizaion을 진행합니다.  
<img src="./images/network.png">
<pre>
<h3> <strong>변경한점</strong> </h3>
1. 조금 더 만화같은 결과를 내기 위하여 KMeans를 이용해 color분포를 단순화하는 preprocessing을 진행했습니다.
2. 모델이 style image로 사용하는 dataset 양 옆에 있는 검은 block을 style로 착각하지 않게끔 하기위하여 
   crop해주었습니다.
   <img src = "./images/before.jpg" width=40%> <img src= "./images/img0.jpg" width=22.5%>
3. 기존의 코드를 그대로 사용하면 style의 content또한 결과에 들어가므로 
   style만 가져오고자하는 목적에 어긋나 style 을 평균을 내주었습니다. 
   이는 그림체가 같은 style image 여러개를 VGG에 넣은 feature값들을
   평균을 내주어 구현하였습니다.
4. 원래 모델은 강한 그림체를(ex 반고흐) 적용하는 것에는 괜찮았으나 
   미국애니메이션과같은 단순한 그림체를 적용하는 것에 성능이 좋지 않아
   실험을 통해 최대한 그럴듯한 결과를 내놓게끔 hyperparameter들을
   수정하였습니다.
5. style feature를 추출할 때에 VGG의 앞쪽 레이어일수록 특이한 무늬를
   생성하는 것을 확인하여 특이한 무늬들에 대한 영향을 줄이고, 뒷 레이어에 
   더 가중치를 두기 위해 아래의 exponential 계수를 추가해주었습니다.
   style_loss의 magnitude가 높아져 style_weight를 더 줄여주었습니다.
   아래와 같이 수정했습니다.
   for i in VGG_layer:
       style_loss += <strong>np.exp(i**4)</strong>(VGG_i(x)
	                   -VGG_i(style))
</pre>


이 과정을 지정한 epoch 만큼 진행을 하면 loss가 점점 줄게되면서 content와 style의 loss 비율에 맞게 x가 변화되며 이 optimization 결과가 x에 나오게됩니다.

</pre>