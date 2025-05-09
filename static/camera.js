let videoStream;
document.addEventListener('DOMContentLoaded', () => {
  const video = document.getElementById('video');
  const canvas = document.getElementById('canvas');
  const preview = document.getElementById('image-preview');
  const captureBtn = document.getElementById('capture-btn');

  if (!video || !canvas || !captureBtn || !preview) return;
  navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
      videoStream = stream;
      video.srcObject = stream;
    })
    .catch(err => {
      alert('Cannot access camera: ' + err.message);
    });

  captureBtn.addEventListener('click', () => {
    console.log("Camera script loaded");
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(blob => {
      console.log("Capturing image from video...");
      const file = new File([blob], 'captured.jpg', { type: 'image/jpeg' });
      console.log("Created blob:", blob);
      preview.src = URL.createObjectURL(file);
      preview.style.display = 'block';

      // Зберігаємо файл у preview для подальшої обробки
      preview.fileBlob = file;
      preview.dataset.fromCamera = "true";
      console.log("Saved to preview:", file);
    }, 'image/jpeg');
  });
});

window.addEventListener('beforeunload', () => {
  if (videoStream) {
    videoStream.getTracks().forEach(track => track.stop());
  }
});
