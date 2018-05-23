--[[
We will use torch to extract features.
This code is used to extract image features.
]]
require 'misc.DataLoader'
require 'torch'
require 'loadcaffe'
require 'nn'
-- local
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'
-- cuda
require 'cutorch'
require 'cunn'
local t = require 'misc.transforms'

------------------------------------------------------------------------------------------
-- prepare data
------------------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Extract features')
cmd:text()
cmd:text('Options')
cmd:option('-dataset', 'refcoco_unc', 'name of our dataset_splitBy')
cmd:option('-cnn_proto', 'models/vgg/VGG_ILSVRC_16_layers_deploy.prototxt', 'path to cnn prototxt')
cmd:option('-cnn_model', 'models/vgg/VGG_ILSVRC_16_layers.caffemodel','path to cnn model')
cmd:option('-batch_size', 50, 'batch of images to be fed into cnn')
cmd:option('-gpuid', 0, 'gpu_id to be used')
cmd:option('-view', 0, 'set to 1 if you have torch-opencv installed and want to check image.')
local opt = cmd:parse(arg)

------------------------------------------------------------------------------------------
-- load cnn model
------------------------------------------------------------------------------------------
-- set gpu
cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed
-- set cudnn to be fastest, I don't like to save memory
local cudnn = require 'cudnn'
cudnn.fastest, cudnn.benchmark = true, true
-- load truncated cnn
--local cnn_raw = loadcaffe.load(opt.cnn_proto, opt.cnn_model, 'cudnn')
--local cnn_fc7 = net_utils.build_cnn(cnn_raw)
--cnn_fc7:evaluate()
--cnn_fc7:cuda()
local resnet_model = torch.load('resnet-200.t7'):cuda()
resnet_model:evaluate()
------------------------------------------------------------------------------------------
-- extract img_feats
------------------------------------------------------------------------------------------
-- DataLoader
local data_json = 'cache/prepro/' .. opt.dataset .. '/data.json'
local data_h5 = 'cache/prepro/' .. opt.dataset .. '/data.h5'
local loader = DataLoader{data_h5 = data_h5, data_json = data_json}

-- Image Directory
local IMAGE_DIR
if string.match(opt.dataset, 'coco') then
	IMAGE_DIR = 'data/images/mscoco/images/train2014'
elseif string.match(opt.dataset, 'clef') then
	IMAGE_DIR = 'data/images/saiapr_tc-12'
else
	print('No image directory prepared for ' .. opt.dataset)
	os.exit()
end

-- Prepare h5 file
local feats_folder = 'cache/feats/' .. opt.dataset
if not utils.file_exists('cache/feats') then os.execute('mkdir cache/feats') end
if not utils.file_exists(feats_folder) then os.execute('mkdir ' .. feats_folder) end
local feats = hdf5.open(feats_folder .. '/img_feats.h5', 'w')

-- extract
local images = loader.images
local img_feats = torch.zeros(#images, 2048):float()

for bs=1, #images, opt.batch_size do
	local be = math.min(bs+opt.batch_size-1, #images)

	local raw_imgs = torch.zeros(be-bs+1, 3, 224, 224):cuda()
	local ib = 1
	for ix = bs, be do
		-- get h5_id
		local h5_id = images[ix]['h5_id']
		assert(h5_id == ix, 'h5_id not match.')
		-- get raw image
		local file_name = images[ix]['file_name']
		local img_path = path.join(IMAGE_DIR, file_name)
		-- make range (1-255)
		--local raw_img = image.load(img_path) * 255
        local raw_img = image.load(img_path, 3, 'float')
        
        local meanstd = {
           mean = { 0.485, 0.456, 0.406 },
           std = { 0.229, 0.224, 0.225 },
        }

        local transform = t.Compose{
           --t.Scale(256),
           t.ColorNormalize(meanstd),
           --t.CenterCrop(224),
        }
        raw_img = transform(raw_img)
        
		-- view if you have torch's opencv installed
		if opt.view == 1 then utils.viewRawImg(raw_img) end
		-- feed into batch
		raw_imgs[ib] = net_utils.prepro_img(raw_img, true)
		ib = ib+1
	end

	-- extract feats
	--img_feats[{ {bs, be}, {} }] = cnn_fc7:forward(raw_imgs):float()
    
    resnet_model:forward(raw_imgs):float()
    
    local output = resnet_model:get(13).output:float()
    
    img_feats[{ {bs, be}, {} }] = output
    
	print(string.format('%s/%s done.', be, #images))
end
feats:write('img_feats', img_feats:float())
print(string.format('img_feats.h5 extracted in %s', feats_folder))













