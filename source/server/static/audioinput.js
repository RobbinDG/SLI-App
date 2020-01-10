const recordAudio = () => {
    return new Promise(resolve => {
        navigator.mediaDevices.getUserMedia({
                audio: {"sampleRate": {"exact": 44100}, "channelCount": 2},
                video: false
            }
        )
            .then(stream => {
                const options = {mimeType: 'audio/webm'};
                const audioChunks = [];
                const mediaRecorder = new MediaRecorder(stream, options);

                mediaRecorder.addEventListener("dataavailable", event => {
                    if (event.data.size > 0)
                        audioChunks.push(event.data);
                });

                const start = () => {
                    mediaRecorder.start();
                };

                const stop = () => {
                    return new Promise(resolve => {
                        mediaRecorder.addEventListener("stop", () => {
                            const audioBlob = new Blob(audioChunks);
                            const audioUrl = URL.createObjectURL(audioBlob);
                            const audio = new Audio(audioUrl);
                            const play = () => {
                                audio.play();
                            };

                            resolve({audioBlob, audioUrl, play, audio});
                        });

                        mediaRecorder.stop();
                    });
                };

                resolve({start, stop});
            });
    });
};

let recorder;

async function startRecording() {
    recorder = await recordAudio();
    recorder.start();
}

async function stopRecording() {
    const audio = await recorder.stop();
    document.getElementById("player").src = (audio.audio.src);
    console.log(audio.audioUrl);
    var link = document.getElementById("download");
    link.href = audio.audioUrl;
    link.download = "aDefaultFileName.ogg";
}