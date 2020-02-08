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

    document.getElementById("classifications").innerText = "Working...";

    document.getElementById("player").src = audio.audio.src;
    console.log(audio.audioUrl);

    let file = blobToFile(audio.audioBlob, "recording.ogg");

    let formdata = new FormData();
    formdata.append("oggfile", file);
    request("/transfer_ogg", formdata, (data) => {
         document.getElementById("classifications").innerHTML = "";
        for (let i = 0; i < 6; ++i) {
            let node = document.createElement("div");
            node.className += "col text-center";

            let lang = document.createElement("p");
            lang.innerText = idxToLanguage(i);
            node.appendChild(lang);

            let prob = document.createElement("p");
            prob.innerText = (data.language[i] * 100).toFixed(2).toString() + "%";
            node.appendChild(prob);

            document.getElementById("classifications").appendChild(node);
        }
    }, () => {});
}

function blobToFile(theBlob, fileName) {
    theBlob.lastModifiedDate = new Date();
    theBlob.name = fileName;
    return theBlob;
}

function idxToLanguage(index) {
    switch (index) {
        case 0: return "Dutch / Nederlands";
        case 1: return "English";
        case 2: return "German / Deutsch";
        case 3: return "French / Francais";
        case 4: return "Italian / Italiano";
        case 5: return "Spanish / Espaniol";
        default: return "Something went wrong: " + index;
    }
}