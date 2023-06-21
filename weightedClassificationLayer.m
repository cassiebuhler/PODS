classdef weightedClassificationLayer < nnet.layer.ClassificationLayer
    
    properties
        % (Optional) Layer properties
        % Vector of weights corresponding to the classes in the training
        % data
        ClassWeights
    end
    
    methods
        function layer = weightedClassificationLayer(classWeights, name)
            % (Optional) Create a myClassificationLayer.
            
            layer.ClassWeights = classWeights;
            
            % Set layer name
            if nargin == 2
                layer.Name = name;
            end
            
            % Set layer description
            layer.Description = 'Weighted cross entropy';
        end
        
        function loss = forwardLoss(layer, Y, T)
            % loss = forwardLoss(layer, Y, T) returns the weighted cross
            % entropy loss between the predictions Y and the training
            % targets T.
            N = size(Y,2);
            Y = squeeze(Y);
            T = squeeze(T);
            W = layer.ClassWeights;
            loss = -sum(W*(T.*log(Y)))/N;
        end
        
        
    end
end


