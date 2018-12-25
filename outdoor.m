net_weights= 'jointF.caffemodel';
net_model = 'DeployT.prototxt';
phase = 'test';

  caffe.set_mode_gpu();
  gpu_id = 0;  % we will use the first gpu in this demo
  caffe.set_device(gpu_id)
net = caffe.Net(net_model, net_weights, phase);
wsz = 15; % window size
 
A=[200,200,200]/255;

infolder='E:\code\dehazingJ\examples';
output='outputnature/'
gamma=1.5;
n_colors=3;
files=dir(fullfile(infolder,'*.png'));
for fileindex=1:length(files)
imname=fullfile(infolder,files(fileindex).name);
im =imread(imname);

[h w   c]=size(im);
net.blobs('data').reshape([h w c 1])
R=zeros(400,400,400);

method='our';
A= Airlight(im, method, wsz);
A=A/255.0;
% A=[0.53, 0.53, 0.53];
%A=[0.8 ,0.9 ,0.94];
%A=[0.67 ,0.72 ,0.825];
%A=[200,200,200]/255;
%A=[0.81 ,0.81 ,0.82];
im=im2double(im);

im=im.^gamma;
img=im;

im(:,:,1)=im(:,:,1)-A(1);
im(:,:,2)=im(:,:,2)-A(2);
im(:,:,3)=im(:,:,3)-A(3);
im=single(im);
% Evaluate network on an image
im_data =single(im);
 t= net.forward({im_data});
transmission_estimation=t{2};








%im=img;
trans_lower_bound = 1 - min(bsxfun(@rdivide,im2double(img),reshape(A,1,1,3)) ,[],3);
transmission_estimation= max(transmission_estimation, trans_lower_bound);
transmission_min=0.1;
transmission=max(transmission_min,transmission_estimation);
r = 15;
r0 = 50;
eps = 10^-3; 

gray_I =rgb2gray(img);
%tt1 = ordfilt2(tt1, 1, ones(r,r), 'symmetric');
%tt1=1-tt1;
transmission= guidedfilter(gray_I, transmission,r0, eps);

img_dehazed = zeros(h,w,n_colors);
leave_haze = 1.0; % leave a bit of haze for a natural look (set to 1 to reduce all haze)
for color_idx = 1:3
    img_dehazed(:,:,color_idx) = ( img(:,:,color_idx) - A(color_idx) )./ max(transmission,transmission_min)+A(color_idx);
end
%for color_idx = 1:3
%   img_dehazed(:,:,color_idx) =  im(:,:,color_idx)./ max(transmission*1.0,transmission_min)+A(color_idx);
      
%end
% Limit each pixel value to the range [0, 1] (avoid numerical problems)
img_dehazed(img_dehazed>1) = 1;
img_dehazed(img_dehazed<0) = 0;
img_dehazed = img_dehazed.^(1/gamma);
adj_percent = [0.005, 0.995];
img_dehazed = adjust(img_dehazed,adj_percent);
%figure, imshow(img_dehazed , []);
% radiometric correction
 name=[output files(fileindex).name];
% adj_percent = [0.005, 0.995];
%img_dehazed = adjust(img_dehazed,adj_percent);
imwrite(img_dehazed ,name);
% For display, we perform a global linear contrast stretch on the output, 
% clipping 0.5% of the pixel values both in the shadows and in the highlights 
%adj_percent = [0.005, 0.995];
%img_dehazed = adjust(img_dehazed,adj_percent);


%figure, imshow(img_dehazed , []);
%imwrite(img_dehazed ,'road1_D1_our5.png');
end
