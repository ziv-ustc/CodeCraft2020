/*
    团队 : 守望&静望&观望
    成绩 ：0.0892 (TOP100)
*/
#include <bits/stdc++.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/shm.h>
#include <sys/ipc.h>
#include <sys/wait.h>

#include <arm_neon.h>

using namespace std;

#define TEST
#define READ_TRAIN_SIZE 800
#define CHAR_NUM_PER_LINE 6002
#define MAX_CHAR_NUM_PER_LINE 7002
#define TRA_NUM_PROCESS 4
#define TES_NUM_PROCESS 14
#define PREDICT_NUM_PROCESS 14

#define Momentum   // stepSize=0.01959  alpha=0.7 beta=0

float stepSize = 0.01f;
float alpha = 0.9f, beta = 0.99f;
float wtInitV = 0.01f;
int maxIterTimes = 1;
float predictTrueThresh = 0.2f;

struct Data {
    Data():label(-1){ }
    float features[1000];
    int label;
};

struct Param {
    vector<float> wtSet;
};

class LR {
public:
    void train();
    void predict();
    LR(string trainFile, string sharedTrainFile, string testFile,string sharedTestFile, string predictOutFile);
private:
    Data* trainDataSet;
    size_t NumTraData;
    int shmidTrain;
    Data* testDataSet;
    size_t NumTesData;
    int shmidTest;
    vector<int> predictVec;
    Param param;
    string trainFile;
    string testFile;
    string sharedTrainFile;
    string sharedTestFile;
    string predictOutFile;
    string weightParamFile = "modelweight.txt";

private:
    bool init();
    bool loadTrainData();
    bool loadTestData();
    inline float vecProduct_neon(float *A, float *B, int *len);
    int storePredict(vector<int> &predict);
    void initParam();
    float wxbCalc(const Data *data);
    float sigmoidCalc(const float wxb);
    double fast_exp(double x) {
        double d;
        *(reinterpret_cast<int*>(&d) + 0) = 0;
        *(reinterpret_cast<int*>(&d) + 1) = static_cast<int>(1512775 * x + 1072632447);
        return d;
    }
private:
    int featuresNum;
};

LR::LR(string trainF, string sharedTrainF, string testF, string sharedTestF, string predictOutF) {
    trainFile = trainF;
    sharedTrainFile = sharedTrainF;
    sharedTestFile = sharedTestF;
    testFile = testF;
    predictOutFile = predictOutF;
    featuresNum = 0;
    init();
}

bool LR::loadTrainData() {
    int fd1 = open(trainFile.c_str(), O_RDONLY);
    if (fd1 == -1) {
        perror("Error opening file for writing");
        exit(EXIT_FAILURE);
    }
    long len = lseek(fd1, 0, SEEK_END);
    NumTraData = READ_TRAIN_SIZE;
    char* buf = (char*)mmap(0, len, PROT_READ, MAP_SHARED, fd1, 0);
    close(fd1);

    shmidTrain = shmget(IPC_PRIVATE, NumTraData*sizeof(Data), IPC_CREAT|0600); // IPC_CREAT|IPC_EXCL
    if (shmidTrain < 0) {
        perror("get shm_ipc error");
        exit(1);
    }
    pid_t pid;
    int patchSize = NumTraData / TRA_NUM_PROCESS;
    int patchSizeMax = len / TRA_NUM_PROCESS;
    for (int i = 1; i < TRA_NUM_PROCESS; ++i) {
        pid = fork();
        if (pid == 0) {
            trainDataSet = (Data*)shmat(shmidTrain, NULL, 0);
            Data* data = trainDataSet + patchSize*i;
            char* buff = buf + patchSizeMax*i;
            int featNum = 0;
            int index = 0;
            long lVal = 0;
            int count = 1;
            while (*buff != '\n') ++buff;
            ++buff;
            while (index < patchSize) {
                if (*(buff + 1) == '\n') {
                    data[index].label = *buff - '0';
                    featNum = 0;
                    ++index;
                    lVal = 0;
                    count = 1;
                    buff += 2;
                } else {
                    if (*buff == '-') {
                        ++buff;
                        while (*buff != '.') {
                            lVal = lVal * 10 + buff[0] - '0';
                            ++buff;
                        }
                        ++buff;
                        while (*buff != ',') {
                            lVal = lVal * 10 + *buff - '0';
                            count *= 10;
                            ++buff;
                        }
                        data[index].features[featNum++] = float(-1 * lVal) / count;
                        lVal = 0;
                        count = 1;
                        ++buff;
                    }
                    else {
                        while (*buff != '.') {
                            lVal = lVal * 10 + buff[0] - '0';
                            ++buff;
                        }
                        ++buff;
                        while (*buff != ',') {
                            lVal = lVal * 10 + *buff - '0';
                            count *= 10;
                            ++buff;
                        }
                        data[index].features[featNum++] = float(lVal) / count;
                        lVal = 0;
                        count = 1;
                        ++buff;
                    }
                }
            }
            shmdt(trainDataSet); // 将本进程和内存区脱离,但并不删除该动态内存
            exit(1);
        }
    }
    trainDataSet = (Data*)shmat(shmidTrain, NULL, 0);
    Data* data = trainDataSet;
    char* buff = buf;
    int featNum = 0;
    int index = 0;
    long lVal = 0;
    int count = 1;
    while (index < patchSize) {
        if (*(buff + 1) == '\n') {
            data[index].label = *buff - '0';
            featNum = 0;
            ++index;
            lVal = 0;
            count = 1;
            buff += 2;
        } else {
            if (*buff == '-') {
                ++buff;
                while (*buff != '.') {
                    lVal = lVal * 10 + buff[0] - '0';
                    ++buff;
                }
                ++buff;
                while (*buff != ',') {
                    lVal = lVal * 10 + *buff - '0';
                    count *= 10;
                    ++buff;
                }
                data[index].features[featNum++] = float(-1 * lVal) / count;
                lVal = 0;
                count = 1;
                ++buff;
            } else {
                while (*buff != '.') {
                    lVal = lVal * 10 + buff[0] - '0';
                    ++buff;
                }
                ++buff;
                while (*buff != ',') {
                    lVal = lVal * 10 + *buff - '0';
                    count *= 10;
                    ++buff;
                }
                data[index].features[featNum++] = float(lVal) / count;
                lVal = 0;
                count = 1;
                ++buff;
            }
        }
    }
    int i = 0;
    while (i < TRA_NUM_PROCESS-1) {
        pid_t pid = wait(NULL);
        if (pid > 0) ++i;
    }
    return true;
}


void LR::initParam() {
    for (int i = 0; i < featuresNum; i++) param.wtSet.push_back(wtInitV);
}

bool LR::init() {
    trainDataSet = NULL;
    bool status = loadTrainData();
    if (status != true) return false;
    featuresNum = 1000;
    param.wtSet.clear();
    initParam();
    return true;
}

inline float LR::wxbCalc(const Data *data) {
    float mulSum = 0.0L, wtv, feav;
    for (int i = 0; i < param.wtSet.size(); i++) {
        wtv = param.wtSet[i];
        feav = data->features[i];
        mulSum += wtv * feav;
    }
    return mulSum;
}

inline float LR::sigmoidCalc(const float wxb) {
    float expv = fast_exp(-1 * wxb);
    float expvInv = 1 / (1 + expv);
    return expvInv;
}

inline float LR::gradientSlope(const Data* dataSet, int index, const vector<float> &sigmoidVec) {
    float gsV = 0.0L,sigv, label;
    for (int i = 0; i < NumTraData; i++) {
        if (trainDataSet[i].label < 0) continue;
        sigv = sigmoidVec[i];
        label = dataSet[i].label;
        gsV += (label - sigv) * (dataSet[i].features[index]);
    }
    gsV = gsV / i;
    return gsV;
}

void LR::train() {
    float sigmoidVal, alpha_t = 1.0L, beta_t = 1.0L;
    vector<float> fM,sM,gD,deltaW;
    gD.resize(param.wtSet.size(),0);
    deltaW = fM = sM = gD;
    float32x4_t alpha_vec = vdupq_n_f32(alpha), nalpha_vec = vdupq_n_f32(1-alpha), step_vec = vdupq_n_f32(stepSize);
    const size_t m = NumTraData;
    for (int k = 0; k < maxIterTimes; ++k) {
        for (size_t i = 0; i < m; ++i) {
            float label = trainDataSet[i].label;
            if (label < 0) continue;
            float wtxVal = 0.0L;
            int len = 1000;
            wtxVal = vecProduct_neon(param.wtSet.data(), &(trainDataSet[i].features[0]), &len);
            sigmoidVal = 1 / (1 +  fast_exp(-1 * wtxVal));
            float32x4_t label_sub_sig = vdupq_n_f32(label - sigmoidVal), grad_vec, feature_vec, fM_vec, wt_vec;
            for (size_t j = 0; j < param.wtSet.size(); j+=4) {
                feature_vec = vld1q_f32(&(trainDataSet[i].features[0])+j);
                fM_vec = vld1q_f32((&fM[0])+j);
                wt_vec = vld1q_f32(param.wtSet.data()+j);
                grad_vec = vmulq_f32(label_sub_sig, feature_vec);
#ifdef Momentum
                fM_vec = vmlaq_f32(vmulq_f32(fM_vec,alpha_vec), nalpha_vec, grad_vec);
                wt_vec = vmlaq_f32(wt_vec, step_vec, fM_vec);
                vst1q_f32(&fM[0]+j, fM_vec);
                vst1q_f32(param.wtSet.data()+j, wt_vec);
#endif
            }
        }
    }
}

void LR::predict() {
    float sigVal;
    int predictVal;
    loadTestData();
    int shmidPredict = shmget(IPC_PRIVATE, NumTesData*sizeof(int), IPC_CREAT|0600);
    if (shmidPredict < 0) {
        perror("get shm_ipc error");
        exit(1);
    }
    pid_t pid;
    int PRE_NUM_PROCESS = PREDICT_NUM_PROCESS;
    int patchSize = NumTesData / PRE_NUM_PROCESS;
    for (int i = 0; i < PRE_NUM_PROCESS-1; ++i) {
        pid = fork();
        if (pid == 0) {
            int* preSet = (int*)shmat(shmidPredict, NULL, 0);
            for (int j = 0; j < patchSize; ++j) {
                if(testDataSet[patchSize*i+j].label!=10) {
                    preSet[patchSize*i+j] = -1;
                    continue;
                }
                float mulSum = 0.0L;
                int len = param.wtSet.size();
                mulSum = vecProduct_neon(param.wtSet.data(), &(testDataSet[patchSize*i+j].features[0]), &len);
                sigVal =  1 / (1+fast_exp(-1 * mulSum));
                predictVal = sigVal >= predictTrueThresh ? 1 : 0;
                preSet[patchSize*i+j] = predictVal;
            }
            shmdt(preSet);
            exit(1);
        }
    }
    int *preSet = (int*)shmat(shmidPredict, NULL, 0);
    for (int j = patchSize*(PRE_NUM_PROCESS-1); j < NumTesData; ++j) {
        if(testDataSet[j].label!=10){
            preSet[j] = -1;
            continue;
        }
        float mulSum = 0.0L;
        int len = param.wtSet.size();
        mulSum = vecProduct_neon(param.wtSet.data(),&(testDataSet[j].features[0]),&len);
        sigVal =  1/(1+fast_exp(-1 * mulSum));
        predictVal = sigVal >= predictTrueThresh ? 1 : 0;
        preSet[j] = predictVal;
    }

    int i = 0;
    while (i < PRE_NUM_PROCESS-1) {
        pid_t pid = wait(NULL);
        if (pid > 0) ++i;
    }

    shmdt(testDataSet);
    shmctl(shmidTest,IPC_RMID,NULL);

    FILE *fptr;
    if ((fptr = fopen(predictOutFile.c_str(), "w")) == NULL) {
        cout << "Error: can not find result.txt !" << endl;
        exit(1);
    }
    for (int i = 0; i < NumTesData; ++i) {
        if (preSet[i] == -1) continue;
        if (preSet[i] == 0) fprintf(fptr, "0\n");
        else fprintf(fptr, "1\n");
    }
    fclose(fptr);
    shmdt(preSet);
    shmctl(shmidPredict, IPC_RMID, NULL);
}

inline float LR::vecProduct_neon(float *A, float *B, int *len) {
    float sum = 0;
    float32x4_t vecRes = vdupq_n_f32(0), vecA, vecB;
    for (int n = 0; n < *len; n+=4) {
        vecA = vld1q_f32(A+n);
        vecB = vld1q_f32(B+n);
        vecRes = vmlaq_f32(vecRes, vecA, vecB);
    }
    float32x2_t res = vadd_f32(vget_high_f32(vecRes), vget_low_f32(vecRes));
    sum += vget_lane_f32(vpadd_f32(res,res), 0);
    return sum;
}

bool LR::loadTestData() {
    shmdt(trainDataSet);
    shmctl(shmidTrain, IPC_RMID,NULL);
    int fd1 = open(testFile.c_str(), O_RDONLY);
    if (fd1 == -1) {
        perror("Error opening file for writing");
        exit(EXIT_FAILURE);
    }
    long len = lseek(fd1, 0, SEEK_END);
    NumTesData = len/5990;
    unsigned char *buf = (unsigned char*)mmap(0, len, PROT_READ, MAP_SHARED, fd1, 0);
    close(fd1);
    shmidTest = shmget(IPC_PRIVATE, NumTesData*sizeof(Data), IPC_CREAT|0600);
    if (shmidTest < 0) {
        perror("get shm_ipc error");
        exit(1);
    }
    pid_t pid;
    long patchSize = NumTesData / TES_NUM_PROCESS;
    long patchSizeMax = len / TES_NUM_PROCESS;
    unsigned char mulg[8] = {1, 0, 100, 10, 1, 0, 1, 0};
    unsigned char mull[8] = {1, 10, 1, 0, 0, 0, 0, 0};
    unsigned short mulls[8] = {1000, 0, 1, 1, 1, 0, 1000, 0};
    uint8x16_t zero_vec = vdupq_n_u8(48);
    for (int i = 1; i < TES_NUM_PROCESS; ++i) {
        pid = fork();
        if (pid == 0) {
            testDataSet = (Data*)shmat(shmidTest, NULL, 0);
            Data* data = testDataSet + patchSize*i;
            if (i < TES_NUM_PROCESS-1) {
                long step = patchSizeMax*(i+1)-1;
                unsigned char *buff = buf + patchSizeMax*i - 1;
                int featNum = 0;
                int index = 0;
                while (*buff != '\n') ++buff;
                ++buff;
                uint8x16_t val_vec;
                uint8x8_t mulg_vec = vld1_u8(&mulg[0]);
                uint8x8_t mull_vec = vld1_u8(&mull[0]);
                uint16x8_t muls_vec = vld1q_u16(&mulls[0]);
                while (1) {
                    uint8x16_t buf_vec = vld1q_u8(buff);
                    val_vec = vsubq_u8(buf_vec, zero_vec);
                    uint16x8_t resG_vec = vmull_u8(mulg_vec, vget_high_u8(val_vec));
                    uint16x8_t resG_vec2 = vmulq_u16(resG_vec, muls_vec);
                    uint16x8_t resL_vec = vmull_u8(mull_vec, vget_low_u8(val_vec));
                    uint32x4_t resEG_vec = vpaddlq_u16(resG_vec2);
                    uint32x4_t resEL_vec = vpaddlq_u16(resL_vec);
                    unsigned int Res4[4], Res5[4];
                    vst1q_u32(&Res4[0], resEG_vec);
                    vst1q_u32(&Res5[0], resEL_vec);
                    data[index].features[featNum++] = float(Res4[0]+Res4[1]+Res4[2]) / 1000;
                    data[index].features[featNum++] = float(Res4[3]+Res5[0]+Res5[1]) / 1000;
                    if (buff[11] == 10) {
                        data[index].label = 10;
                        ++index;
                        featNum = 0;
                        if (&buff[11] >= buf+step) break;
                    }
                    buff += 12;
                }
            } else {
                long step = len - 1;
                unsigned char *buff = buf + patchSizeMax*i - 1;
                int featNum = 0;
                int index = 0;
                while (*buff != '\n') ++buff;
                ++buff;
                uint8x16_t val_vec;
                uint8x8_t mulg_vec = vld1_u8(&mulg[0]);
                uint8x8_t mull_vec = vld1_u8(&mull[0]);
                uint16x8_t muls_vec = vld1q_u16(&mulls[0]);
                while (1) {
                    uint8x16_t buf_vec = vld1q_u8(buff);
                    val_vec = vsubq_u8(buf_vec,zero_vec);
                    uint16x8_t resG_vec = vmull_u8(mulg_vec,vget_high_u8(val_vec));
                    uint16x8_t resG_vec2 = vmulq_u16(resG_vec,muls_vec);
                    uint16x8_t resL_vec = vmull_u8(mull_vec,vget_low_u8(val_vec));
                    uint32x4_t resEG_vec = vpaddlq_u16(resG_vec2);
                    uint32x4_t resEL_vec = vpaddlq_u16(resL_vec);
                    unsigned int Res4[4], Res5[4];
                    vst1q_u32(&Res4[0], resEG_vec);
                    vst1q_u32(&Res5[0], resEL_vec);
                    data[index].features[featNum++] = float(Res4[0]+Res4[1]+Res4[2]) / 1000;
                    data[index].features[featNum++] = float(Res4[3]+Res5[0]+Res5[1]) / 1000;
                    if (buff[11] == 10) {
                        data[index].label = 10;
                        ++index;
                        featNum = 0;
                        if (&buff[11] >= buf+step) break;
                    }
                    buff += 12;
                }
            }
            shmdt(testDataSet);
            exit(1);
        }
    }
    testDataSet = (Data*)shmat(shmidTest, NULL, 0);
    Data* data = testDataSet;
    unsigned char* buff = buf;
    int featNum = 0;
    int index = 0;
    long step = patchSizeMax-1;
    uint8x16_t val_vec;
    uint8x8_t mulg_vec = vld1_u8(&mulg[0]);
    uint8x8_t mull_vec = vld1_u8(&mull[0]);
    uint16x8_t muls_vec = vld1q_u16(&mulls[0]);
    while (1) {
        uint8x16_t buf_vec = vld1q_u8(buff);
        val_vec = vsubq_u8(buf_vec,zero_vec);
        uint16x8_t resG_vec = vmull_u8(mulg_vec,vget_high_u8(val_vec));
        uint16x8_t resG_vec2 = vmulq_u16(resG_vec,muls_vec);
        uint16x8_t resL_vec = vmull_u8(mull_vec,vget_low_u8(val_vec));
        uint32x4_t resEG_vec = vpaddlq_u16(resG_vec2);
        uint32x4_t resEL_vec = vpaddlq_u16(resL_vec);
        unsigned int Res4[4], Res5[4];
        vst1q_u32(&Res4[0], resEG_vec);
        vst1q_u32(&Res5[0], resEL_vec);
        data[index].features[featNum++] = float(Res4[0]+Res4[1]+Res4[2]) / 1000;
        data[index].features[featNum++] = float(Res4[3]+Res5[0]+Res5[1]) / 1000;
        if (buff[11] == 10) {
            data[index].label = 10;
            featNum = 0;
            ++index;
            if(&buff[11] >= buf+step) break;
        }
        buff += 12;
    }
    int i = 0;
    while (i < TES_NUM_PROCESS-1) {
        pid_t pid = wait(NULL);
        if (pid > 0) ++i;
    }
    return true;
}

bool loadAnswerData(string awFile, vector<int> &awVec) {
    ifstream infile(awFile.c_str());
    if (!infile) {
        cout << "打开答案文件失败" << endl;
        exit(0);
    }
    while (infile) {
        string line;
        int aw;
        getline(infile, line);
        if (line.size() > 0) {
            stringstream sin(line);
            sin >> aw;
            awVec.push_back(aw);
        }
    }
    infile.close();
    return true;
}

int LR::storePredict(vector<int> &predict) {
    const int size = predictVec.size();
    FILE *fptr;
    if ((fptr = fopen(predictOutFile.c_str(), "w")) == NULL) {
        cout << "Error: can not find result.txt !" << endl;
        exit(1);
    }
    for (int i = 0; i < size; ++i) {
        if (predict[i] == 0) fprintf(fptr, "0\n");
        else fprintf(fptr, "1\n");
    }
    fclose(fptr);
    return 0;
}

int main() {
    vector<int> answerVec;
    vector<int> predictVec;
    int correctCount;
    float accurate;
    string trainFile = "/data/train_data.txt";
    string sharedTraindFile = "/data/shared_traindata.txt";
    string testFile = "/data/test_data.txt";
    string sharedtestFile = "/data/shared_testdata.txt";
    string predictFile = "/projects/student/result.txt";
    string answerFile = "/projects/student/answer.txt";
#ifdef TEST
    trainFile = "../data/train_data.txt";
    sharedTraindFile = "../data/shared_traindata.txt";
    testFile = "../data/test_data.txt";
    sharedtestFile = "../data/shared_testdata.txt";
    predictFile = "../projects/student/result.txt";
    answerFile = "../projects/student/answer.txt";
    auto startTime = std::chrono::steady_clock::now();
#endif
    LR logist(trainFile, sharedTraindFile, testFile, sharedtestFile, predictFile);
#ifdef TEST
    auto endTime = std::chrono::steady_clock::now();
    float part1_ds = std::chrono::duration<float>(endTime-startTime).count();
    cout << "loading traindata time: " << part1_ds << "s." << endl;
    startTime = std::chrono::steady_clock::now();
#endif
    logist.train();
#ifdef TEST
    endTime = std::chrono::steady_clock::now();
    float part2_ds = std::chrono::duration<float>(endTime-startTime).count();
    cout << "training time: " << part2_ds << "s." << endl;
    loadAnswerData(answerFile, answerVec);
    startTime = std::chrono::steady_clock::now();
#endif
    logist.predict();
#ifdef TEST
    endTime = std::chrono::steady_clock::now();
    float part3_ds = std::chrono::duration<float>(endTime-startTime).count();
    cout << "loading testdata and predicting time: " << part3_ds << "s." << endl;
    cout << "total time: " << part1_ds+part2_ds+part3_ds << "s." << endl;;
    loadAnswerData(predictFile, predictVec);
    cout << "test data set size is " << predictVec.size() << endl;
    correctCount = 0;
    for (int j = 0; j < predictVec.size(); j++) {
        if (j < answerVec.size()) {
            if (answerVec[j] == predictVec[j]) correctCount++;
        } else {
            cout << "answer size less than the real predicted value" << endl;
        }
    }
    accurate = ((float)correctCount) / answerVec.size();
    cout << "the prediction accuracy is " << accurate << endl;
#endif
    return 0;
}

