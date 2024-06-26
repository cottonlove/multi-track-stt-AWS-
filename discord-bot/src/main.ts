const Eris = require("eris");
const fs = require("fs");
const wavConverter = require("wav-converter");
const path = require("path");
// const doSTT = require('./stt').default;
const doTranslation = require('./translate').default;
const { Client } = require('@elastic/elasticsearch')
require('dotenv').config();
var spawn = require('child_process').spawn;

const bot = new Eris(process.env.DISCORD_BOT_TOKEN, {
    getAllUsers: true,
    intents: 98303	
});
const es = new Client({
    cloud: {
        id: process.env.ELASTIC_CLOUD_ID
      },
      auth: {
        username: process.env.ELASTIC_USERNAME,
        password: process.env.ELASTIC_PASSWORD
      }
})

const Constants = Eris.Constants;

const SENTENCE_INTERVAL = 1500; 

const channelMap = new Map();
// const userVoiceDataMap = new Map();
// const memberMap = new Map();
// const channelGame = "LOL";
// const ttsQueue = [];

function stereoToMono(stereoBuffer) {
    const numChannels = 2;
    const bytesPerSample = 2;

    const totalSamples = stereoBuffer.length / (numChannels * bytesPerSample);

    const monoBuffer = Buffer.alloc(totalSamples * bytesPerSample);

    for (let i = 0; i < totalSamples; i++) {
        const leftIndex = i * numChannels * bytesPerSample;
        const rightIndex = leftIndex + bytesPerSample;

        const leftValue = stereoBuffer.readInt16LE(leftIndex);
        const rightValue = stereoBuffer.readInt16LE(rightIndex);

        const averageValue = Math.round((leftValue + rightValue) / 2);

        monoBuffer.writeInt16LE(averageValue, i * bytesPerSample);
    }

    return monoBuffer;
}

// // Define a function to transcribe audio
// function transcribeAudio(filename, language, samplerate, channelGame) {
//     return new Promise((resolve, reject) => {
//         const pythonProcess = spawn('python', ['src/transcribe.py', `outputs/${filename}.wav`, language, samplerate, channelGame]);

//         let transcriptionResult = '';

//         // Listen for data on the stdout stream
//         pythonProcess.stdout.on('data', function(data) {
//             transcriptionResult += data.toString();
//         });

//         // Handle errors
//         pythonProcess.stderr.on('data', function(data) {
//             reject(`Error: ${data}`);
//         });

//         // Handle process exit
//         pythonProcess.on('close', function(code) {
//             if (code !== 0) {
//                 reject(`Python process exited with code ${code}`);
//             } else {
//                 console.log('Python process completed successfully.');
//                 resolve(transcriptionResult.trim());
//             }
//         });
//     });
// }


//code for 48kHz audio  
bot.on("ready", () => {
    console.log("Ready!");
    setInterval(() => {
        channelMap.forEach((channel) => {
            const userVoiceDataMap = channel.userVoiceDataMap;
            const memberMap = channel.memberMap;
            const channelGame = channel.channelGame;
            const ttsQueue = channel.ttsQueue;
            // console.log(userVoiceDataMap);
            // console.log(memberMap);
            // console.log(channelGame);
            // console.log(ttsQueue);

            userVoiceDataMap.forEach((userData, userID) => {
                const currentTime = Date.now();
                const elapsedTimeSinceLastSTT = currentTime - userData.startTime;
                const samplerate = 48000
                if (currentTime - userData.lastTime >= SENTENCE_INTERVAL || elapsedTimeSinceLastSTT >= 15000 ) {
                    const filename = userData.filename;

                    const inputFilePath = `./outputs/${filename}.pcm`;
                    // const outputFilePath = `./outputs/${filename}-mono.pcm`;

                    // const stereoBuffer = fs.readFileSync(inputFilePath);

                    // const monoBuffer = stereoToMono(stereoBuffer);

                    // fs.writeFileSync(outputFilePath, monoBuffer);

                    const pcmData_stereo = fs.readFileSync(`./outputs/${filename}.pcm`)

                    const wavData_stereo = wavConverter.encodeWav(pcmData_stereo, {
                        numChannels: 2,
                        sampleRate: samplerate,
                        byteRate: 16
                    });
                    fs.writeFileSync(`./outputs/${filename}_stereo.wav`, wavData_stereo);

                    // const pcmData = fs.readFileSync(`./outputs/${filename}-mono.pcm`)
                    // const wavData = wavConverter.encodeWav(pcmData, {
                    //     numChannels: 1,
                    //     sampleRate: samplerate,
                    //     byteRate: 16
                    // });
    
                    // fs.writeFileSync(`./outputs/${filename}.wav`, wavData);
                    // console.log("write wav file");

                    const memberData = memberMap.get(userID);
                    ttsQueue.push({filename, text: "", name: memberData.name, language: memberData.language.split("-")[0], finish: false});
                    // doSTT(filename, memberData.language, samplerate, channelGame) //STT on mono-pcm file
                    // .then(({filename, text}) => {
                        // const fileIndex = ttsQueue.findIndex((item) => item.filename === filename);
                        // ttsQueue[fileIndex].text = text;
                        // ttsQueue[fileIndex].finish = true;
                    // })

                    // Call the transcribeAudio function
                    // transcribeAudio(filename, memberData.language, samplerate, channelGame)
                    // .then((text) => {
                    //     console.log('Transcription Result:', text);
                    //     const fileIndex = ttsQueue.findIndex((item) => item.filename === filename);
                    //     ttsQueue[fileIndex].text = text;
                    //     ttsQueue[fileIndex].finish = true;
                    // })
                    // .catch((error) => {
                    //     console.error(error);
                    // });

                    //2. spawn을 통해 "python 파이썬파일.py" 명령어 실행
                    const result = spawn('python', ['src/transcribe.py', `outputs/${filename}_stereo.wav`]);
                    result.stdout.on('data', function(data) {
                        console.log(data.toString());
                        const fileIndex = ttsQueue.findIndex((item) => item.filename === filename);
                        ttsQueue[fileIndex].text = data.toString();
                        ttsQueue[fileIndex].finish = true;
                        console.log('translate call')
                    })

                    console.log('DONE');
                    // Handle errors
                    result.stderr.on('data', function(data) {
                        console.error(`Error: ${data}`);
                    });
                    userVoiceDataMap.delete(userID);
                    fs.unlink(`./outputs/${filename}.pcm`, () => {});
                    console.log('HI DONE');
                    // fs.unlink(`./outputs/    ${filename}-mono.pcm`, () => {});
                }
            });

            if (ttsQueue.length > 0 && ttsQueue[0].finish) {
                const { filename, text, name, language, result } = ttsQueue.shift();
                console.log(text, name, language);
                if (text !== "") {
                    const translationPromises = ['de', 'ko', 'en'].map(targetLanguage => {
                        if (language !== targetLanguage) {
                            return doTranslation(text, language, targetLanguage, channelGame);
                        } else {
                            return {TargetLanguageCode: targetLanguage, TranslatedText: text};
                        }
                    });
                    
                    Promise.all(translationPromises)
                    .then((results) => {
                        results.forEach(result => {
                            console.log(result.TargetLanguageCode, result.TranslatedText);
                            memberMap.forEach((user) => {
                                if (user.language.split("-")[0] === result.TargetLanguageCode) {

                                    if (name !== user.name) {
                                        bot.getDMChannel(user.id).then((channel) => {
                                            channel.createMessage(`${name} : ${result.TranslatedText}`);}
                                        )
                                    }

                                }
                            });
                        });

                        es.index({
                            index: 'discord_game_stt_translation',
                            body: {
                                "english_sentence": results[2].TranslatedText,
                                "category": channelGame,
                                "korean_sentence": results[1].TranslatedText,
                                "turkish_sentence": results[0].TranslatedText,
                                "source_language": language,
                            }
                        }).then((res) => {
                            console.log(res);
                        }).catch((err) => {
                            console.log(err);
                        })
                    })

                }
            }
        })
    }, SENTENCE_INTERVAL);
});


bot.on("messageCreate", (msg) => {
    if(msg.content === "!ping") {
        bot.createMessage(msg.channel.id, "Pong!");
    } else if (msg.content == "!language"){
        bot.createMessage(msg.channel.id, {
            content: "Choose your language!",
            components: [
                {
                    type: Constants.ComponentTypes.ACTION_ROW,
                    components: [
                        {   type: Constants.ComponentTypes.BUTTON,
                            style: Constants.ButtonStyles.PRIMARY,
                            custom_id: "ko-KR",
                            label: "한국어",
                            disabled: false
                        },
                        {
                            type: Constants.ComponentTypes.BUTTON,
                            style: Constants.ButtonStyles.PRIMARY,
                            custom_id: "en-US",
                            label: "English",
                            disabled: false
                        },
                        {
                            type: Constants.ComponentTypes.BUTTON,
                            style: Constants.ButtonStyles.PRIMARY,
                            custom_id: "de-DE",
                            label: "Deutsch",
                            disabled: false
                        }
                    ]
                }
            ]
        }); 
    } else if (msg.content === "!game") {
        bot.createMessage(msg.channel.id, {
            content: "Choose your game!",
            components: [
                {
                    type: Constants.ComponentTypes.ACTION_ROW,
                    components: [
                        {   type: Constants.ComponentTypes.BUTTON,
                            style: Constants.ButtonStyles.PRIMARY,
                            custom_id: "LOL",
                            label: "League of Legends",
                            disabled: false
                        },
                        {
                            type: Constants.ComponentTypes.BUTTON,
                            style: Constants.ButtonStyles.PRIMARY,
                            custom_id: "overwatch",
                            label: "Overwatch",
                            disabled: false
                        },
                        {
                            type: Constants.ComponentTypes.BUTTON,
                            style: Constants.ButtonStyles.PRIMARY,
                            custom_id: "AmongUs",
                            label: "Among Us",
                            disabled: false
                        },
                        {
                            type: Constants.ComponentTypes.BUTTON,
                            style: Constants.ButtonStyles.PRIMARY,
                            custom_id: "pubg",
                            label: "Battlegrounds",
                            disabled: false
                        }
                    ]
                }
            ]
        });
    } else if (msg.content === "!join") {
        if (!msg.member.voiceState.channelID) {
            bot.createMessage(msg.channel.id, "You are not in a voice channel.");
            return;
        } else {
            bot.joinVoiceChannel(msg.member.voiceState.channelID).catch((err) => {
                bot.createMessage(msg.channel.id, "Error joining voice channel: " + err.message);
                console.log(err);
            }).then((voiceConnection) => {
                bot.createMessage(msg.channel.id, "hello");
                channelMap.set(msg.member.voiceState.channelID, { 
                    userVoiceDataMap: new Map(),
                    memberMap: new Map(),
                    channelGame: "LOL",
                    ttsQueue: []
                });
                const channel = channelMap.get(msg.member.voiceState.channelID);
                const userVoiceDataMap = channel.userVoiceDataMap;
                const memberMap = channel.memberMap;
                bot.getChannel(msg.member.voiceState.channelID).voiceMembers.forEach((member) => {
                    if (!memberMap.has(member.id) && !member.bot)
                        memberMap.set(member.id, {
                        id: member.id,
                        name: member.username,
                        language: "en-US"
                    });
                })
                const voiceReceiver = voiceConnection.receive("pcm")
                voiceReceiver.on("data", (voiceData, userID, timestamp, sequence) => {
                    if (userID) {
                        const currentTime = Date.now();
                        if (!userVoiceDataMap.has(userID)) {
                            userVoiceDataMap.set(userID, {
                                streams: fs.createWriteStream(`./outputs/${userID}-${currentTime}.pcm`),
                                lastTime: currentTime,
                                startTime: currentTime,
                                filename: `${userID}-${currentTime}`
                            });
                        }
                        const userVoiceData = userVoiceDataMap.get(userID);
                        userVoiceData.streams.write(voiceData);
                        userVoiceData.lastTime = currentTime;
                    }
                })
            })
        } 
    } else if (msg.content === "!leave") {
        if (!msg.member.voiceState.channelID) {
            bot.createMessage(msg.channel.id, "You are not in a voice channel.");
            return;
        } else {
            channelMap.delete(msg.member.voiceState.channelID);
            bot.leaveVoiceChannel(msg.member.voiceState.channelID)
            bot.createMessage(msg.channel.id, "bye");
        }
    } else if (msg.content == "!getLanguageSettings") {
        let languageSettings = "";
        const channel = channelMap.get(msg.member.voiceState.channelID);
        if (channel === undefined) {
            bot.createMessage(msg.channel.id, "Bot is not in a voice channel.");
            return;
        }
        const memberMap = channel.memberMap;
        memberMap.forEach((user) => {
            languageSettings += `${user.name} : ${user.language}\n`;
        });
        
        if (languageSettings === "") {
            bot.createMessage(msg.channel.id, "No user has set the language yet.");
        } else {
            bot.createMessage(msg.channel.id, languageSettings);
        }
    } else if (msg.content == "!getGameSettings") {
        const channel = channelMap.get(msg.member.voiceState.channelID);
        if (channel === undefined) {
            bot.createMessage(msg.channel.id, "Bot is not in a voice channel.");
            return;
        }
        const channelGame = channel.channelGame;
        bot.createMessage(msg.channel.id, `The game is set to ${channelGame}.`);
    }
});

bot.on("voiceChannelJoin", (member, newChannel) => {
    const channel = channelMap.get(member.voiceState.channelID);
    if (channel === undefined) {
        return null;
    }
    const memberMap = channel.memberMap;
     if (!memberMap.has(member.id) && !member.bot)
        memberMap.set(member.id, {
            id: member.id,
            name: member.username,
            language: "en-US" //default language is English
        });
});

bot.on("voiceChannelLeave", (member, newChannel) => {
    const channel = channelMap.get(member.voiceState.channelID);
    if (channel === undefined) {
        return null;
    }
    const memberMap = channel.memberMap;
    if (memberMap.has(member.id) && !member.bot)
       memberMap.delete(member.id);
});

bot.on("interactionCreate", (interaction) => {
    const channel = channelMap.get(interaction.member.voiceState.channelID);
    if (channel === undefined) {
        return interaction.createMessage("Bot is not in a voice channel.");
    }
    const memberMap = channel.memberMap;
    if(interaction instanceof Eris.ComponentInteraction) { 
        if (["LOL", "overwatch", "AmongUs", "pubg"].includes(interaction.data.custom_id)) {
            channel.channelGame = interaction.data.custom_id;
            return interaction.createMessage({
                content: `${interaction.data.custom_id} is set.`
            })
        } else {
            const userId = interaction.member.user.id;
            const userLanguage = interaction.data.custom_id;

            if (memberMap.has(userId)) {
                const user = memberMap.get(userId);
                user.language = userLanguage; 
                memberMap.set(userId, user);
            } else {
                return interaction.createMessage({
                        content: "Please select a language after the bot enters the voice channel."
                })
            }

            if(userLanguage === "ko-KR") {
                return interaction.createMessage({
                        content: `<@${userId}> 한국어로 설정되었습니다.` 
                })
            } else if (userLanguage === "en-US") {
                return interaction.createMessage({
                    content: `<@${userId}> English is set.`    
                })
            } else if (userLanguage === "de-DE") {
                return interaction.createMessage({
                    content: `<@${userId}> Auf Deutsch eingestellt.`
                })
            }
        }
    }
});

bot.connect();
