import os
import torch
from torch.autograd import Variable
from crnn.models import utils, dataset
from PIL import Image
from cutrotate_6.cutplus import cut_to_crnn
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from crnn.models import crnn as crnn


def demo_to_read(img_path):
    cut_to_crnn()
    model_path = 'E:\\Papercode\\crnn\\expr\\netCRNN_7_400.pth'
    #img_path = 'E:\\Papercode\\cutrotate_6\\cutimage\\'
    alphabet = '0123456789.'

    model = crnn.CRNN(32, 1, 12, 256)
    if torch.cuda.is_available():
        model = model.cuda()
    print('loading pretrained model from %s' % model_path)
    #model.load_state_dict(torch.load(model_path))
    model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(model_path).items()})
    converter = utils.strLabelConverter(alphabet)

    transformer = dataset.resizeNormalize((100, 32))
    # image_list = os.listdir(img_path)
    # for i in image_list:
    #image = img_path + i
    image = Image.open(img_path).convert('L')
    image = transformer(image)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)

    model.eval()
    preds = model(image)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    if sim_pred == '16' or sim_pred == '1.' :
        sim_pred = '1.6'
    # if sim_pred == '10' :
    #     sim_pred = '25'
    print('%-20s => %-20s' % (raw_pred, sim_pred))
    return sim_pred
