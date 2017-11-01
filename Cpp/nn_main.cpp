#include<iostream>
#include<vector>
#include<cstdlib>
#include<cmath>

using namespace std;

class Neuron;

typedef vector<Neuron> NNLayer;


struct Connection
{
    double weight;
    double deltaWeight;
};
/***************************** Class Neuron ***************************/

class Neuron
{
public:
    static double eta;
    static double alpha;
    static double activationFun(double input);
    static double activationFunDerivative(double input);

    Neuron(int numOp, int index);
    void feedForward(NNLayer &prevLayer);
    void setOutVal(double outVal) { m_outVal = outVal;}
    double getOutVal() {return m_outVal;}

    void calOutLyrGradient(double outVal);
    void calHiddenLyrGradient(NNLayer &nextLayer);

    double sumDOW(NNLayer &nextLayer);

    void updateInputWeights(NNLayer& prevLayer);
private:


    double m_outVal;
    vector<Connection> m_weights;
    int my_Index;
    double m_gradient;
};

double Neuron::alpha = 0.15;
double Neuron::eta = 0.5;
/***************************** Class NeuralNetwork ***************************/

class NeuralNet {

public:
    NeuralNet(vector<int>&arch);

    static double randomWeight() { return rand()/ double (RAND_MAX); }
    void forwardPass(const vector<double> &inputVal);
    void backwordPass(const vector<double> &outputVal);
    void getResult(vector<double> &result);

private:
    double m_error;
    vector<NNLayer> m_layers;
};

void showVector(vector<double> &inputVal) {
    for( int i = 0; i < inputVal.size(); i++) {
        cout<<" "<<inputVal[i];
    }
    cout<<endl;
}


int main() {

    vector<int> architecture;

    architecture.push_back(2);
    architecture.push_back(4);
    architecture.push_back(1);

    NeuralNet myNet(architecture);

    for(int epoc = 1; epoc < 2 ; epoc++) {
        cout<<endl<< " Pass " << epoc << endl;

        // 1,1 = 0
        vector<double> inputVal;
        inputVal.push_back(1);
        inputVal.push_back(1);
        cout<<" Input :: ";
        showVector(inputVal);

        myNet.forwardPass(inputVal);

        vector<double> resultVal;
        myNet.getResult(resultVal);
        cout<<" Result :: ";
        showVector(resultVal);

        vector<double>targetVal;
        targetVal.push_back(1);
        cout<<" Expected :: ";
        showVector(targetVal);

        myNet.backwordPass(targetVal);

        cout<<endl;

        // 0,0 = 0
        inputVal.clear();
        inputVal.push_back(0);
        inputVal.push_back(0);
        cout<<" Input :: ";
        showVector(inputVal);

        myNet.forwardPass(inputVal);

        resultVal.clear();
        myNet.getResult(resultVal);
        cout<<" Result :: ";
        showVector(resultVal);

        targetVal.clear();
        targetVal.push_back(0);
        cout<<" Expected :: ";
        showVector(targetVal);

        myNet.backwordPass(targetVal);

        cout<<endl;

        // 1,0 = 1
        inputVal.clear();
        inputVal.push_back(1);
        inputVal.push_back(0);
        cout<<" Input :: ";
        showVector(inputVal);

        myNet.forwardPass(inputVal);

        resultVal.clear();
        myNet.getResult(resultVal);
        cout<<" Result :: ";
        showVector(resultVal);

        targetVal.clear();
        targetVal.push_back(1);
        cout<<" Expected :: ";
        showVector(targetVal);

        myNet.backwordPass(targetVal);

        cout<<endl;

        // 0,1 = 1
        inputVal.clear();
        inputVal.push_back(0);
        inputVal.push_back(1);
        cout<<" Input :: ";
        showVector(inputVal);

        myNet.forwardPass(inputVal);

        resultVal.clear();
        myNet.getResult(resultVal);
        cout<<" Result :: ";
        showVector(resultVal);

        targetVal.clear();
        targetVal.push_back(1);
        cout<<" Expected :: ";
        showVector(targetVal);

        myNet.backwordPass(targetVal);

    }

}

NeuralNet::NeuralNet(vector<int> &arch)
{
    int num_layers = arch.size();

    for(int l_num = 0; l_num < num_layers; l_num++) {
        m_layers.push_back(NNLayer());

        //if op layer then there are no further connections.
        int numOp = (l_num == arch.size() - 1 ) ? 0 : arch[l_num + 1];

        //create all neurons including bias
        for(int num_neuron = 0 ; num_neuron <= arch[l_num]; num_neuron++) {
            m_layers.back().push_back(Neuron(numOp, num_neuron));
            cout<<" created Neuron"<<endl;
        }

        //force bias neuron output to 1
        m_layers.back().back().setOutVal(1.0);
    }

}

void NeuralNet::forwardPass(const vector<double> &inputVal)
{
    //load inputs
    for(int i = 0 ; i < inputVal.size(); i ++) {
        m_layers[0][i].setOutVal(inputVal[i]);
    }

    //forward feed
    for(int layer_num = 1; layer_num < m_layers.size(); layer_num++) {
        NNLayer & prevLayer = m_layers[layer_num - 1];
//        cout<< "forwardPass :: layer = "<< layer_num;
        // access all but bias neurons
        for(int n = 0; n < m_layers[layer_num].size() - 1; n++) {
            m_layers[layer_num][n].feedForward(prevLayer);
        }
    }
}

void NeuralNet::backwordPass(const vector<double> &outputVal)
{
    // calculate error of network
    NNLayer& outLayer = m_layers.back();
    m_error = 0;

    // loop all neurons except the bias neuron
    for(int n = 0; n < outLayer.size() - 1; n++) {
        double delta = outputVal[n] - outLayer[n].getOutVal();
        m_error += delta * delta;
    }

    m_error /= outLayer.size() - 1; // get average seuared error
    m_error = sqrt(m_error);

    // calculate gradient of output layer

    // loop all neurons except the bias neuron
    for(int n = 0; n < outLayer.size() - 1; n++) {
        outLayer[n].calOutLyrGradient(outputVal[n]);
    }

    // calculate gradient of hidden layers
    for(int lyr_num = m_layers.size() - 2; lyr_num > 0; -- lyr_num){
        NNLayer & hiddenLayer = m_layers[lyr_num];
        NNLayer & nextLayer = m_layers[lyr_num + 1];

        // loop all neurons except the bias neuron
        for(int n = 0; n < outLayer.size() - 1; n++) {
            hiddenLayer[n].calHiddenLyrGradient(nextLayer);
        }
    }

    // update weights according to the gradient
    for(int lyr_num = m_layers.size() - 1; lyr_num > 0; -- lyr_num){
        NNLayer & currentLayer = m_layers[lyr_num];
        NNLayer & prevLayer = m_layers[lyr_num - 1];

        // loop all neurons except the bias neuron
        for(int n = 0; n < currentLayer.size() - 1; n++) {
            currentLayer[n].updateInputWeights(prevLayer);
        }
    }
}

void NeuralNet::getResult(vector<double> &result)
{
    result.clear();

    for(int n = 0 ; n < m_layers.back().size() -1 ; n++) {
        result.push_back(m_layers.back()[n].getOutVal());
    }
}

double Neuron::activationFun(double input)
{
    return tanh(input);
}

double Neuron::activationFunDerivative(double input)
{
    return (1.0 - (input*input));
}

Neuron::Neuron(int numOp, int index)
{
    my_Index = index;

    for(int c =0; c < numOp; c++) {
        m_weights.push_back(Connection());

        m_weights.back().weight = NeuralNet::randomWeight();

    }
}

void Neuron::feedForward(NNLayer& prevLayer)
{
    double sum = 0.0;

    // Including the boas neuron in the previous layer
    for(int n = 0; n < prevLayer.size(); n++) {
        double val = prevLayer[n].getOutVal();
        sum += ( val * prevLayer[n].m_weights[my_Index].weight );
    }

    m_outVal = Neuron::activationFun(sum);
}

void Neuron::calOutLyrGradient(double outVal)
{
    double delta = outVal - m_outVal;
    m_gradient = delta * Neuron::activationFunDerivative(m_outVal);
}

void Neuron::calHiddenLyrGradient(NNLayer &nextLayer)
{
    double dow = sumDOW(nextLayer);
    m_gradient = dow * Neuron::activationFun(m_outVal);
}

double Neuron::sumDOW(NNLayer &nextLayer)
{
    double sum = 0;
    // sum our contributions of the errors at the nodes we feed
    for(int n = 0 ; n < nextLayer.size() -1 ; n++) {
        sum += m_weights[n].weight * nextLayer[n].m_gradient;
    }

    return sum;
}

void Neuron::updateInputWeights(NNLayer &prevLayer)
{
    // all neurons including the bias
    for(int n = 0 ; n < prevLayer.size() ; n++) {
        Neuron & currNeuron = prevLayer[n];
        double oldDeltaWeight = currNeuron.m_weights[my_Index].deltaWeight;

        double newDeltaWeight =
                // Individual input, magnified by the gradient and train rate
                eta
                * currNeuron.getOutVal()
                * m_gradient
                // Also add a momentum = a fraction of the previous delta weight
                + alpha
                * oldDeltaWeight;

        currNeuron.m_weights[my_Index].deltaWeight = newDeltaWeight;
        currNeuron.m_weights[my_Index].weight += newDeltaWeight;

    }
}
