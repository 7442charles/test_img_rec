const imageUpload = document.getElementById('imageUpload')
const imagePreview = document.getElementById('imagePreview')
const spinner = document.getElementById('spinner')

Promise.all([
    faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
    faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
    faceapi.nets.ssdMobilenetv1.loadFromUri('/models')
]).then(start)

async function start() {
    spinner.style.display = 'block' // show the spinner
    const labeledFaceDescriptors = await loadLabeledImages()
    const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6)
    spinner.style.display = 'none' // hide the spinner
    let image
    let canvas
    document.body.append('Loaded')
    imageUpload.addEventListener('change', async () => {
        if (image) image.remove()
        if (canvas) canvas.remove()
        image = await faceapi.bufferToImage(imageUpload.files[0])
        imagePreview.innerHTML = '' // clear the previous image
        imagePreview.appendChild(image)
        canvas = faceapi.createCanvasFromMedia(image)
        imagePreview.appendChild(canvas)
        const displaySize = { width: image.width, height: image.height }
        faceapi.matchDimensions(canvas, displaySize)
        const detections = await faceapi.detectAllFaces(image).withFaceLandmarks().withFaceDescriptors()
        const resizedDetections = faceapi.resizeResults(detections, displaySize)
        const results = resizedDetections.map(d => faceMatcher.findBestMatch(d.descriptor))
        results.forEach((result, i) => {
            alert(`Face : ${result.label}  Accuracy : ${result.distance.toFixed(2)}`);
        })
    })
}

async function loadLabeledImages() {
    const labels = ['Trump', 'Captain America', 'Captain Marvel', 'Khalif', 'Obama', 'Thor', 'Tony Stark', 'Chao'];
    
    return Promise.all(
        labels.map(async label => {
            const descriptions = [];
            let i = 1;
            
            while (true) {
                try {
                    // Try loading both .jpg and .png images
                    let img;
                    try {
                        img = await faceapi.fetchImage(`https://raw.githubusercontent.com/7442charles/test_img_rec/main/labeled_images/${label}/${i}.jpg`);
                    } catch (error) {
                        img = await faceapi.fetchImage(`https://raw.githubusercontent.com/7442charles/test_img_rec/main/labeled_images/${label}/${i}.png`);
                    }
                    
                    const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor();
                    descriptions.push(detections.descriptor);
                    i++;
                } catch (error) {
                    // Exit the loop if no more images are found
                    break;
                }
            }

            return new faceapi.LabeledFaceDescriptors(label, descriptions);
        })
    );
}