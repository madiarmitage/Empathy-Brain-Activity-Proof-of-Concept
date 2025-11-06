lut = readtable('COMP NEURO BIDS\BN_Atlas_246_LUT.txt')

info = niftiinfo('COMP NEURO BIDS/BN_Atlas_246_1mm.nii');
raw_atlas = niftiread(info);

atlas = double(raw_atlas) * info.MultiplicativeScaling + info.AdditiveOffset;

unique(atlas(:))
%%
insula_indices = 165:174
insula_mask = ismember(atlas, insula_indices);
%%
datafolder = ('C:\Users\madis\Documents\COMP NEURO BIDS')

participant_ids = [1,2,3,4,5,6,7,8,14,18,31,34,39,43,50,52,53,57,58,63,64,67,71,74,75,76,81,88,102,106,109,115,119,122,130,136,139,149,154,156,158,159,167,169,179,180,187,191,192,193,194,195,196,197]
runs = 2;
insula_values = zeros(length(participant_ids), 1);

for positionnmbr = 1:length(participant_ids)
    sub_id = participant_ids(positionnmbr);
    run_values = zeros(runs,1);

    for r = 1:runs
        brainfile = fullfile(datafolder, ...
            sprintf('sub-%03d', sub_id), 'func', ...
            sprintf('sub-%03d_task-empathy_acq-MNI152NLin6Asym_rec-preproc_run-%02d_bold.nii', sub_id, r));
        braininfo = (niftiinfo(brainfile));
        brain_activity = double(niftiread(braininfo));
        insula_voxels = brain_activity(insula_mask);
        run_values(r) = mean(insula_voxels);
    end
    insula_values(positionnmbr) = mean(run_values);

end
%%
disp(insula_values)
summary(insula_values)
%%
histogram(insula_values)
xlabel('Mean insula activation')
ylabel('Number of participants')
%%
n = 54;
meanTEQ = 46.25;
sdTEQ = 7.55;
empathy = round(min(max(meanTEQ + sdTEQ * randn (54,1),0),64))
summary(empathy)
brain_activity = insula_values

histogram(empathy)
xlabel('Empathy')
ylabel('Number of Participants')
title ('Distribution of Empathy Scores')

scatter(brain_activity, empathy, "filled")
xlabel('Brain Activity')
ylabel('Empathy Scores')
title('Relationship Between Brain Activity and Empathy')
lsline

model = fitlm(brain_activity, empathy)

predicted_empathy = predict(model, brain_activity);

scatter(brain_activity, empathy, "filled")
hold on
plot(brain_activity, predicted_empathy, 'r', 'LineWidth', 2)
hold off
xlabel('Brain Activity')
ylabel('Empathy')
title('Model Predictions (Red Line)')

%%
n = 54;
meanTEQ = 45;
sdTEQ = 7; 
empathy = round(min(max(meanTEQ + sdTEQ * randn(54,1),0),64))
brain_activity = insula_values

scatter(brain_activity, empathy, 'filled')
xlabel('Brain Activity')
ylabel('Empathy')
title('Brain Activity vs Empathy')

high_empathy = empathy >= 45;
sum(high_empathy)
sum(~high_empathy)
log_model = fitglm(brain_activity, high_empathy, 'Distribution','binomial')

predict_prob = predict(log_model, brain_activity)
summary(predict_prob)

classify_predict = predict_prob >= 0.5

conf_matrix = confusionmat(high_empathy, classify_predict);
accuracy = sum(classify_predict == high_empathy) / length(high_empathy);

disp(conf_matrix)
disp(['Accuracy = ' num2str(accuracy)])