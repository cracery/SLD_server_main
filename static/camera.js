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
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(blob => {
      const file = new File([blob], 'captured.jpg', { type: 'image/jpeg' });

      preview.src = URL.createObjectURL(file);
      preview.style.display = 'block';

      // Зберігаємо файл у preview для подальшої обробки
      preview.fileBlob = file;
      preview.dataset.fromCamera = "true";
    }, 'image/jpeg');
  });
});

window.addEventListener('beforeunload', () => {
  if (videoStream) {
    videoStream.getTracks().forEach(track => track.stop());
  }
});