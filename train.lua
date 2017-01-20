require("hdf5")
require("nn")
require("rnn")
require("nngraph")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', 'qa16.hdf5', 'data file')

-- Hyperparameters
cmd:option('-eta',0.01,'learning rate hyperparameter for lr/nn')
cmd:option('-max_grad_norm',40, 'max norm for RNN models')
cmd:option('-grad_norm','global', '[global, local, off] gradient renormalization')
cmd:option('-N',100,'num epochs hyperparameter for lr/nn')
cmd:option('-D0',20,'num outputs of lookup layer of nn')
cmd:option('-debug','','specify model to debug')
cmd:option('-unit_test',false,'whether run all unit tests')


-- Mem NN Model parameters
cmd:option('-k',3,'num hops for MemNN Model')
cmd:option('-max_history',25,'max history')
cmd:option('-max_sent_len',20,'max sent len')
cmd:option('-pe',false,'enable position encoding')
cmd:option('-te',false,'enable temporal encoding')

-- LSTM parameters
cmd:option('-save',false,'whether to save model')
cmd:option('-saveminacc',0.5,'minimum accuracy on test set required to save model')
cmd:option('-cuda',false,'whether to use cuda')

function main()
  -- Parse input params
  opt = cmd:parse(arg)
  load()
  if opt.debug ~= '' then
    debugModel()
  else
    print(string.format("Using data file: %s", opt.datafile))
    if opt.cuda then
      require("cunn")
      print("Using Cuda")
    else
      print("Using CPU")
    end
    if opt.pe then
      print('Using Position Encoding')
    end
    if opt.te then
      print('Using Temporal Encoding')
    end
    runMemNN()
  end
end


function runMemNN()
  trainMemNN()
  testModel(false)
end

function trainMemNN()
  print(string.format(
    "Training MemN2N model with N = %d, D0 = %d, Eta = %f, "..
    "Max History = %d, Max Sentence Length = %d, Weight Tying = adjacent, "..
    "Max Gradient Norm = %d, Gradient Norm Type = %s, Number of hops = %d",
    opt.N, opt.D0, opt.eta, opt.max_history, opt.max_sent_len,
    opt.max_grad_norm, opt.grad_norm, opt.k))

  D0 = opt.D0
  local eta = opt.eta
  local trainLoss = torch.zeros(opt.N)
  local timer = torch.Timer()

  create_embedding_layers()

  -- other containers
  -- finding match between query & input
  P = nn.Sequential():add(nn.MM(false,true)):add(nn.Squeeze()):add(nn.View(1,opt.max_history)) -- for softmax over u^T m
  P1 = P:clone()
  P2 = P:clone()

  O = nn.MM() -- weighted average of C & P
  O1 = nn.MM()
  O2 = nn.MM()

  -- second layer
  U_next = nn.Sequential()
  U_next:add(nn.CAddTable())

  -- third layer
  U1_next = nn.Sequential()
  U1_next:add(nn.CAddTable())

  W_linear = nn.Linear(D0,nwords,false)
  W = nn.Sequential():add(nn.CAddTable()):add(W_linear):add(nn.LogSoftMax())
  W:add(nn.Squeeze())

  -- nngraph based model
  x_inp = nn.Identity()():annotate({name = 'x', description = 'memories'})
  q_inp = nn.Identity()():annotate({name = 'q', description  = 'query'})
  te_inp = nn.Identity()():annotate({name = 'te_inp', description = 'temporal encoding'})

  x_pre = ltx(x_inp):annotate({name = 'x_pre', description = 'pre embeddings'})
  q_pre = ltq(q_inp):annotate({name = 'q_pre', description = 'pre embeddings'})
  m_pre = A(x_pre):annotate({name = 'm_pre', description = 'memory embeddings'})
  u = B(q_pre):annotate({name = 'u', description = 'query embeddings'})
  c_pre = C(x_inp):annotate({name = 'c_pre', description = 'output embeddings'})
  m = A_mask({m_pre,te_inp}):annotate({name = 'm', description = 'memory embeddings'})
  c = C_mask({c_pre,te_inp}):annotate({name = 'c', description = 'output embeddings'})
  p1 = P({u,m}):annotate({name = 'p1', description = 'distribution over u^T m'})
  o = O({p1,c}):annotate({name = 'o', description = 'p1 weighted c'})

  if opt.k > 1 then
    u1 = U_next({u,o}):annotate({name = 'u1', description = 'u1 layer (u0 + o0)'})

    p2 = P1({u1,c}):annotate({name = 'p2', description = 'distribution over u1^T m'})
    c2_pre = C2(x_inp):annotate({name = 'c2_pre', description = 'output embeddings'})
    c2 = C2_mask({c2_pre,te_inp}):annotate({name = 'c2', description = 'output embeddings'})
    o1 = O1({p2,c2}):annotate({name = 'o1', description = 'p2 weighted c'})

    if opt.k == 3 then
      u2 = U1_next({u1,o1}):annotate({name = 'u2', description = 'u2 layer (u1 + o1)'})
      p3 = P2({u2,c2}):annotate({name = 'p3', description = 'distribution over u2^T m'})
      c3_pre = C3(x_inp):annotate({name = 'c3_pre', description = 'output embeddings'})
      c3 = C3_mask({c3_pre,te_inp}):annotate({name = 'c3', description = 'output embeddings'})
      o2 = O2({p3,c3}):annotate({name = 'o3', description = 'p3 weighted c3'})

      W_linear:share(C3:get(1), 'weight', 'gradWeight')
    end
  end

  if opt.k == 1 then
    a_hat = W({u,o}):annotate({name = 'a_hat', description = 'output'})
  elseif opt.k == 2 then
    a_hat = W({u1,o1}):annotate({name = 'a_hat', description = 'output'})
  else
    a_hat = W({u2,o2}):annotate({name = 'a_hat', description = 'output'})
  end
  model = nn.gModule({x_inp,q_inp,te_inp},{a_hat})

  params, gradParams = model:getParameters()
  crit = nn.ClassNLLCriterion()

  gradParams:zero()
  for i=1, params:size(1) do params[i] = torch.randn(1)[1]/ torch.sqrt(10) end

  if opt.pe then
    resetPE()
  end

  for iEpoch=1, opt.N do
    local idxs = torch.randperm(train_stories:size(1)):long()
    for l=1, train_stories:size(1) do
      zeroLookupTable()

      i = idxs[l]

      local x = getStory(train_stories,i,opt.max_history, opt.max_sent_len)
      local q = train_questions[i]
      local preds = model:forward({x,q,te_mask})

      if opt.unit_test then
        -- embedding rows corresponding to padding should be zero
        nng_x = get_module_output('x')
        nng_m = get_module_output('m')
        nng_c = get_module_output('c')
        nng_c2 = get_module_output('c2')
        nng_c3 = get_module_output('c3')
        n_pad_rows = torch.sum(nng_x[{{},1}]:eq(idx_pad))
        if nng_m then assert(n_pad_rows == n_zero_rows(nng_m), 'Embeddings incorrect.') end
        if nng_c then assert(n_pad_rows == n_zero_rows(nng_c), 'Embeddings incorrect.') end
        if nng_c2 then assert(n_pad_rows == n_zero_rows(nng_c2), 'Embeddings incorrect.') end
        if nng_c3 then assert(n_pad_rows == n_zero_rows(nng_c3), 'Embeddings incorrect.') end
      end

      -- if l == 1 then
      --   local debug_names = Set { "x", "m", "c", "c2", "c3" }
      --   for i,v in ipairs(model.forwardnodes) do
      --     if debug_names[v.data.annotations.name] then
      --       print(v.data.annotations.name)
      --       print(v.data.module.output)
      --     end
      --   end
      -- end

      local max_prob, max_idx = torch.max(preds,1)
      local loss = crit:forward(preds,train_answers[i])
      trainLoss[iEpoch] = trainLoss[iEpoch] + loss
      local dLdinp = crit:backward(preds,train_answers[i])
      gradParams:zero()
      model:backward({x,q,te_mask},dLdinp)
      zeroPEGrad()
      adjustGrad()
      params:add(-eta, gradParams)
    end

    print("epoch "..iEpoch,"loss "..trainLoss[iEpoch])
    print("Total time taken",timer:time().real)

    if iEpoch == 20 then
      print('Re-inserting softmax to P')
      print('Re-inserting softmax to P1')
      print('Re-inserting softmax to P2')
      P:insert(nn.SoftMax(), 3)
      P1:insert(nn.SoftMax(), 3)
      P2:insert(nn.SoftMax(), 3)
      print(P)
      print(P1)
      print(P2)
    end

    testModel(true)

    eta = adjustEta(eta, iEpoch)
  end
end

function create_embedding_layers()
  ltx = nn.LookupTable(nwords,D0)
  ltq = ltx:clone('weight', 'gradWeight')
  A = nn.Sequential()
  B = nn.Sequential()

  if opt.pe then -- position encoding
    A_pe = nn.CMul(opt.max_history,opt.max_sent_len,D0)
    A:add(A_pe)
  end
  A:add(nn.Sum(2))
  if opt.te then -- temporal encoding
    A:add(nn.View(1,D0*opt.max_history)):add(nn.Add(D0*opt.max_history))
      :add(nn.View(opt.max_history,D0))
  end

  -- for query
  -- adding position encoding
  if opt.pe then
    B_pe = nn.CMul(train_questions:size(2),D0)
    B:add(B_pe)
  end
  B:add(nn.Sum(1)):add(nn.View(1,D0))

  -- for output representation of the memories
  C = nn.Sequential():add(nn.LookupTable(nwords,D0))
  if opt.pe then -- position encoding
    C_pe = nn.CMul(opt.max_history,opt.max_sent_len,D0)
    C:add(C_pe)
  end
  C:add(nn.Sum(2))
  if opt.te then -- temporal encoding
    C:add(nn.View(1,D0*opt.max_history)):add(nn.Add(D0*opt.max_history))
      :add(nn.View(opt.max_history,D0))
  end
  
  C2 = C:clone()
  C3 = C:clone()
  if opt.pe then
    C2_pe = C2:get(2)
    C3_pe = C3:get(2)
  end

  A_mask  = nn.CMulTable()
  C_mask  = nn.CMulTable()
  C2_mask = nn.CMulTable()
  C3_mask = nn.CMulTable()

  te_mask = torch.ones(opt.max_history, D0)
end

function adjustGrad()
  if opt.grad_norm == 'off' then
    return
  end
  local grad
  if opt.grad_norm == 'global' then
    renorm(gradParams, opt.max_grad_norm)
  elseif opt.grad_norm == 'local' then
    for i,node in ipairs(model.forwardnodes) do
      if node.data.module then
        local lmod = node.data.module
        if lmod.gradWeight and lmod.gradBias then
          grad = nn.JoinTable(1):forward({lmod.gradWeight:view(-1,1), lmod.gradBias:view(-1,1)})
        elseif lmod.gradWeight then
          grad = lmod.gradWeight:view(-1,1)
        elseif lmod.gradBias then
          grad = lmod.gradBias:view(-1,1)
        end
        renorm(grad, opt.max_grad_norm)
      end
    end
  end
  if opt.unit_test then -- test grad renorm
    newParams, newGradParams = model:getParameters()
    assert(torch.sum(gradParams:eq(newGradParams)) == gradParams:size()[1], 'renorm failed test')
    params = newParams
    gradParams = newGradParams
  end
end

function renorm(t, norm)
  if t and #t:size() > 0 then
    local t_norm = torch.sqrt(t:dot(t))
    local shrinkage = norm / t_norm
    if shrinkage < 1 then
      t:mul(shrinkage)
    end
  end
end

function adjustEta(eta, epoch)
  if epoch <= 100 then
    if epoch % 25 == 0 then
      return eta / 2
    end
  end
  return eta
end

function debugModel()
  local nanswers = test_answers:size(1)
  local all_predictions = torch.zeros(nanswers)
  local all_prediction_scores = torch.zeros(nanswers, nwords)
  local all_marginals

  debugFile = opt.debug
  proto = torch.load(opt.debug)
  model = proto.model
  if proto.options then opt = proto.options end
  max_history = opt.max_history
  D0 = opt.D0
  max_sent_len = opt.max_sent_len
  nstates = max_history + 1
  T = 3
  te_mask = torch.ones(max_history, D0)

  local outDebugFile = hdf5.open(''..debugFile..'.debug', 'w')

  all_marginals = torch.zeros(nanswers, T, opt.max_history)

  for i = 1, test_stories:size(1) do
    local x = getStory(test_stories, i, opt.max_history, opt.max_sent_len)
    local q = test_questions[i]
    all_prediction_scores[i] = model:forward({x,q,te_mask})
    _, all_predictions[i] = torch.max(all_prediction_scores[i]:float(), 1)
    for _, p in ipairs(model.forwardnodes) do
      local hop = -1
      if p.data.annotations.name == 'p1' then hop = 1
      elseif p.data.annotations.name == 'p2' then hop = 2
      elseif p.data.annotations.name == 'p3' then hop = 3
      end
      if hop > -1 then
        all_marginals[i][hop] = p.data.module.output
      end
    end
  end

  print('Accuracy = '..torch.eq(all_predictions:long(), test_answers):sum()/nanswers)
  outDebugFile:write('scores', all_prediction_scores)
  outDebugFile:write('answers', test_answers:squeeze())
  outDebugFile:write('predictions', all_predictions:long())
  outDebugFile:write('marginals', all_marginals)
  outDebugFile:close()
end

function testModel(useHeldout)
  if useHeldout then
    testX = heldout_stories
    testQ = heldout_questions
    testA = heldout_answers
    accuracyText = "Accuracy on held out = "
  else
    testX = test_stories
    testQ = test_questions
    testA = test_answers
    accuracyText = "Accuracy on test set = "
    model = bestHeldoutModel
  end

  if opt.cuda then
    testX = testX:cuda()
    testQ = testQ:cuda()
  end

  local Y_hat = torch.zeros(testA:size(1))
  zeroLookupTable()
  for i=1, testX:size(1) do
    local x = getStory(testX,i,opt.max_history, opt.max_sent_len)
    local q = testQ[i]
    local preds = model:forward({x,q,te_mask})
    _, Y_hat[i] = torch.max(preds:float(),1)
  end
  local correct = torch.eq(Y_hat:long() - testA, 0):sum()
  local accuracy = correct/Y_hat:size(1)
  print(accuracyText..accuracy)

  if useHeldout then
    if bestHeldoutAccuracy < accuracy then
      bestHeldoutAccuracy = accuracy
      bestHeldoutModel = model:clone()
    end
  else
    -- re-test the best model on held out set for sanity check
    local Y_hat_heldout = torch.zeros(heldout_answers:size(1))
    for i=1, heldout_stories:size(1) do
      local x = getStory(heldout_stories,i,opt.max_history, opt.max_sent_len)
      local q = heldout_questions[i]
      local preds = model:forward({x,q,te_mask})
      _, Y_hat_heldout[i] = torch.max(preds:float(),1)
    end
    local accuracy_heldout = torch.eq(Y_hat_heldout:long()-heldout_answers, 0):sum() / Y_hat_heldout:size(1)
    if accuracy_heldout ~= bestHeldoutAccuracy then
      print('Best model on held out set is lost, cannot reproduce accuracy '..
        bestHeldoutAccuracy .. ', actual accuracy = ' .. accuracy_heldout)
    else
      print('Using model which achieved ' .. accuracy_heldout .. ' on held out set.')
    end
    if opt.save and accuracy >= opt.saveminacc then
      local acc = torch.LongTensor({accuracy*10000}):double()[1]/100
      local modelFile = opt.datafile.."."..acc..".memnn"
      torch.save(modelFile, {model = model, options = opt})
      print('Saved model to ' .. modelFile)
    end
  end
end

function makePosEncMat(inputLayer)
  if inputLayer == nil then
    return
  end
  
  local input = inputLayer.weight
  input:zero()

  if input:dim() == 3 then
    num_sent , sent_len, embed_size = input:size(1), input:size(2), input:size(3)
    for i=1, num_sent do
      for j=1, sent_len do
        for k=1, embed_size do
          input[i][j][k] = (1-j/sent_len) - (k/embed_size)*(1- (2*j/sent_len))
        end
      end
    end
  else
    sent_len, embed_size = input:size(1), input:size(2)
    for j=1, sent_len do
      for k=1, embed_size do
        input[j][k] = (1-j/sent_len) - (k/embed_size)*(1- (2*j/sent_len))
      end
    end
  end
end

-- convert 1 x all_stories_by_question to num_stories x max_single_story_len
function getStory(X,q_id,max_history,max_sent_len)
  local story = X[ {q_id, {num_history - max_history + 1, num_history}, {1, max_sent_len} } ]
  
  -- detect empty memories and clear out theta potentials
  local num_empty_sentences = torch.sum(story[{{},1}]:eq(idx_pad))

  if te_mask then
    te_mask:fill(1)
    if num_empty_sentences > 0 then
      te_mask[{{1,num_empty_sentences}}]:fill(0)
    end
  end

  return story
end

function get_module_output(name)
  for i,v in ipairs(model.forwardnodes) do
    if v.data.annotations.name == name and v.data.module then
      return v.data.module.output
    end
  end
  return nil
end

function n_zero_rows(ts)
  if ts then
    return torch.sum(torch.sum(torch.abs(ts),2):eq(0))
  else
    return 0
  end
end

function resetPE()
  makePosEncMat(A_pe)
  makePosEncMat(B_pe)
  makePosEncMat(C_pe)
  makePosEncMat(C2_pe)
  makePosEncMat(C3_pe)
end

function zeroPEGrad()
  if opt.pe then
    if A_pe then A_pe.gradWeight:zero() end
    if B_pe then B_pe.gradWeight:zero() end
    if C_pe then C_pe.gradWeight:zero() end
    if C2_pe then C2_pe.gradWeight:zero() end
    if C3_pe then C3_pe.gradWeight:zero() end
  end
end

function zeroLookupTable()
  zeroWeight(ltx.weight)
  zeroWeight(ltq.weight)
  if ltn then
    zeroWeight(ltn.weight)
  end
  zeroWeight(C.modules[1].weight)
  if C2 then zeroWeight(C2.modules[1].weight) end
  if C3 then zeroWeight(C3.modules[1].weight) end
end

function zeroWeight(wt)
  wt[idx_pad]:zero()
  wt[idx_start]:zero()
  wt[idx_end]:zero()
  wt[idx_rare]:zero()
end

function Set (list)
  local set = {}
  for _, l in ipairs(list) do set[l] = true end
  return set
end

function writeToFile(obj,f)
  local myFile = hdf5.open(f, 'w')
  for k,v in pairs(obj) do
    myFile:write(k, v)
  end
  myFile:close()
end

function load()
  if opt.debug ~= '' then
    opt.datafile = string.sub(opt.debug, 1, 9)
    print('Loading detected data file ' .. opt.datafile)
  end
  -- get the data out of the datafile
  local f = hdf5.open(opt.datafile, 'r')
  local data = f:all()

  idx_start = data.idx_start[1]
  idx_end   = data.idx_end[1]
  idx_pad   = data.idx_pad[1]
  idx_rare  = data.idx_rare[1]

  nwords = data.nwords[1]

  train_stories   = data.train_stories:long() -- [# Questions x Max Story Length]
  train_questions = data.train_questions:long() -- [# Questions x Max Q Length]
  train_answers   = data.train_answers:long() -- [# Questions x 1]
  train_facts     = data.train_facts:long() -- [# Questions x Max Fact Length]

  local ntrains = train_stories:size(1)
  local endtrain = math.floor(ntrains * 0.9)

  heldout_stories = train_stories[{ {endtrain + 1, ntrains} }]
  heldout_questions = train_questions[{ {endtrain + 1, ntrains} }]
  heldout_answers = train_answers[{ {endtrain + 1, ntrains} }]
  heldout_facts = train_facts[{ {endtrain + 1, ntrains} }]

  train_stories = train_stories[{ {1, endtrain} }]
  train_questions = train_questions[{ {1, endtrain} }]
  train_answers = train_answers[{ {1, endtrain} }]
  train_facts = train_facts[{ {1, endtrain} }]  

  test_stories   = data.test_stories:long()
  test_questions = data.test_questions:long()
  test_answers   = data.test_answers:long()
  test_facts     = data.test_facts:long()

  num_history = train_stories:size(2)
  len_sentence = train_stories:size(3)

  opt.max_history = math.min(opt.max_history, num_history)
  opt.max_sent_len = math.min(opt.max_sent_len, len_sentence)

  bestHeldoutAccuracy = 0
end


main()
