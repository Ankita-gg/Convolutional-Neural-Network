package network;

import java.util.ArrayList;
import java.util.List;

import layer.ConvolutionLayer;
import layer.FullyConnectedLayers;
import layer.Layers;
import layer.MaxPoolLayer;

public class NetworkBuilder {

    private NeuralNetwork net;
    private int _inputRows;
    private int _inputCols;
    private double _scaleFactor;
    List<Layers> _layers;

    public NetworkBuilder(int _inputRows, int _inputCols, double _scaleFactor) {
        this._inputRows = _inputRows;
        this._inputCols = _inputCols;
        this._scaleFactor = _scaleFactor;
        _layers = new ArrayList<>();
    }

    public void addConvolutionLayer(int numFilters, int filterSize, int stepSize, double learningRate, long SEED){
        if(_layers.isEmpty()){
            _layers.add(new ConvolutionLayer(filterSize, stepSize, 1, _inputRows, _inputCols, SEED, numFilters, learningRate));
        } else {
            Layers prev = _layers.get(_layers.size()-1);
            _layers.add(new ConvolutionLayer(filterSize, stepSize, prev.getoutputlength(), prev.getoutputrows(), prev.getoutputcolumns(), SEED, numFilters, learningRate));
        }
    }

    public void addMaxPoolLayer(int windowSize, int stepSize){
        if(_layers.isEmpty()){
            _layers.add(new MaxPoolLayer(stepSize, windowSize, 1, _inputRows, _inputCols));
        } else {
            Layers prev = _layers.get(_layers.size()-1);
            _layers.add(new MaxPoolLayer(stepSize, windowSize, prev.getoutputlength(), prev.getoutputrows(), prev.getoutputcolumns()));
        }
    }

    public void addFullyConnectedLayer(int outLength, double learningRate, long SEED){
        if(_layers.isEmpty()) {
            _layers.add(new FullyConnectedLayers(_inputCols*_inputRows, outLength, SEED, learningRate));
        } else {
            Layers prev = _layers.get(_layers.size()-1);
            _layers.add(new FullyConnectedLayers(prev.getoutputelements(), outLength, SEED, learningRate));
        }

    }

    public NeuralNetwork build(){
        net = new NeuralNetwork(_layers, _scaleFactor);
        return net;
    }

}
