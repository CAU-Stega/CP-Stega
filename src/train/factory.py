import train.maml as maml
import train.regular as regular
import train.regular_t as regular_t
import train.finetune as finetune
import train.maml_t as maml_t
import train.maml_lstm as maml_lstm


def train(train_data, val_data, model, args):
   if args.maml:
        if args.embedding == 'lstmatt':
             return maml_lstm.train(train_data, val_data, model, args)
        else:
             return maml.train(train_data, val_data, model, args)
   else:
        return regular.train(train_data, val_data, model, args)


def test(test_data, model, args, verbose=True):

    if args.maml:
        return maml_t.test(test_data, model, args, verbose)
    elif args.mode == 'finetune':
        return finetune.test(test_data, model, args, verbose)
    else:
        return regular_t.test(test_data, model, args, verbose)
