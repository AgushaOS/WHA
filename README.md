# Wavelet Hierarchical Audio 

**WHA (Wavelet Hierarchical Audio)** is an audio codec based on WPT (Wavelet Packet Transform), written in C++, with two main goals:
1) **High-speed audio compression** for music and film audio
2) **An excellent speed-to-quality ratio**, delivering high audio quality at all available bitrates while being very fast to encode and decode.

There are two modes for encoding: **speed** and **quality**.
More details about both modes provided below.

## Codec Specifications

**Quality mode:**
- Default mode for encoding
- Encoding speed is 50-66% of MP2's encoding speed on stereo audio for all bitrates
- Decoding speed is 50-66% of MP2's decoding speed on stereo audio for all bitrates
- Excellent audio quality at bitrates 128 kbps and higher
- Good to very good audio quality at 96 kbps
- Good audio quality at 64 kbps
- Handles spectrally sparse audio well
- Recommended for bitrates below 128 kbps due to its lower noise levels

**Speed mode:**
- Encoding speed is 95-105% of MP2's encoding speed on stereo audio for all bitrates
- Decoding speed is 95-105% of MP2's decoding speed on stereo audio for all bitrates
- Not default mode for encoding
- Excellent audio quality at bitrates 128 kbps and higher
- Good to very good audio quality at 96 kbps (less stable)
- Normal to good audio quality at 64 kbps
- Does not handle spectrally sparse music as well
- Not recommended for bitrates below 128 kbps due to its higher noise levels

## PEAQ, ViSQOL and all other perceptual evaluators
- gstPEAQ is better for 128 kbps and higher, but there are some tracks that are problematic for gstPEAQ to evaluate. But it is also good for lower bitrates. Recommended for general use.
- ViSQOL is better for bitrates lower than 128 kbps, but it is really bad for noisy audio (there some tracks, that are graded better in "speed mode" than in "quality mode" on 64 kbps. They've been tested, it isn't true). Recommended for general use only with something else (like gstPEAQ).
- PQEvalAudio has worse correlation with reality than gstPEAQ, but there are tracks that are tested by it better. Not recommended for general use.
- gstPEAQ Advanced is worse of all above in terms of correlation with reality. Not recommended for general use.

## Comparing quality to other codecs
- Quality on all bitrates is better than Vorbis at mostly all music + film data if to judge from tested files
- Quality on all bitrates is better than LC3 at mostly all music + film data if to judge from tested files
- 64 kbps, in terms of quality, are comparable to 128 kbps of MP3
- 96 kbps, in terms of quality, are comparable to 160 kbps of MP3
- 128 kbps, in terms of quality, are comparable to 192+ kbps of MP3

It must be said, that codec was developed for complex music with many details, so anything simple like pure castanets, can sound awful on low bitrates. But castanets + guitar, for example, will be much better. 

For more detailed tests try to look at releases. There are always PEAQ tests there with full table

## Algorithm
**Quality mode:**
- Encoding:
  - Block complexity: **O(NlogN)** 
  - Block memory complexity: **O(N)**

- Decoding:
  - Block complexity: **O(NlogN)**
  - Block memory complexity: **O(N)**

**Speed mode:**
- Encoding:
  - Block complexity: **O(N)** 
  - Block memory complexity: **O(N)**

- Decoding:
  - Block complexity: **O(N)**
  - Block memory complexity: **O(N)**

Details will be published in papers about the codec.

## Usage
**Compiling:**
For usage, it is better to compile before FFTW with AVX optimizations for full potential speed on quality mode.
```bash
//Compiling codec
./setup.sh
```
**Encoding:**
```bash
//Encoding quality mode
./code [input.wav] [bitrate]

//Encoding speed mode
./code [input.wav] [bitrate] --speed

// Both files will be written into [input].wha format
```
**Decoding:**
```bash
//Decoding (for both quality and speed mode)
./decode [input.wha] [output.wav] 
```
