#include "mfcc.h"

int main{
    //25ms = 400 samples
    int frame_len = 400;
    int num_filters = 40;

    KWS::MFCC * kws_feature_ = new KWS::MFCC(13, 400, 8, "povey", false);  //13不用管，8不用管
    float * audio = new float[frame_len];
    float * fbank_out = new float[num_filters];

    kws_feature_->fbank_compute(audio, true , fbank_out);

    if(kws_feature_) delete kws_feature_;
    kws_feature_ = nullptr;
    if(audio) delete audio;
    audio = nullptr;
    if(fbank_out) delete fbank_out;
    fbank_out = nullptr;
}