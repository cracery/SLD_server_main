var videoStream;
document.addEventListener('DOMContentLoaded',function() {
  var video= document.getElementById('video');
  var canvas=document.getElementById('canvas');
  var preview=document.getElementById('image-preview');
  var captureBtn=document.getElementById('capture-btn');
  if (!video || !canvas || !captureBtn || !preview) return;
  navigator.mediaDevices.getUserMedia({video:true})
    .then(function(stream) {
      videoStream= stream;
      video.srcObject=stream;
    })
    .catch(function(err){
      alert('Cannot access camera: '+ err.message);
    });
  captureBtn.addEventListener('click',function() {
    console.log("Camera script loaded");
    canvas.width= video.videoWidth;
    canvas.height= video.videoHeight;    
    var ctx= canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(function(blob) {
      console.log("Capturing image from video...");
      var file= new File([blob],'captured.jpg',{ type: 'image/jpeg' });
      console.log("Created blob:",blob);
      preview.src= URL.createObjectURL(file);
      preview.style.display= 'block';
      // save to preview
      preview.fileBlob= file;
      preview.dataset.fromCamera = "true";
      console.log("Saved to preview:", file);
    }, 'image/jpeg');
  });
});
window.addEventListener('beforeunload',function(){
  if (videoStream) {
    videoStream.getTracks().forEach(function(track){
      track.stop();
    });
  }
});
