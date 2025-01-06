import pickle

import numpy as np


def get_road_adj(filename):
    with open(filename, 'rb') as fo:
        result = pickle.load(fo)
    return result

if __name__ == '__main__':
    r = r'rawstate.pkl'
    # rd = r'rawstate_d.pkl'
    rela = r'roadnet_relation.pkl'
    r_data = get_road_adj(r)
    # rd_data = get_road_adj(rd)
    rela_data = get_road_adj(rela)
    print(type(rela_data))
    with open('r.txt', 'w+') as fo:
        for i in r_data:
            for j in i:
                fo.write(str(j))
                fo.write('\n')
            fo.write('++++++++++++++')

    # with open('rd.txt', 'w+') as fo:
    #     for i in rd_data:
    #         for j in i:
    #             fo.write(str(j))
    #             fo.write('\n')
    #         fo.write('++++++++++++++')

    # with open('rela.txt', 'w+') as fo:
    #     for i in rela_data:
    #         print(i)
    #         fo.writelines(rela_data[i])

    # with open('rela.txt', 'w+') as fo:
    #     for key,value in rela_data:
    #         fo.write(value)
    #         fo.write('\n')
    #         fo.write('++++++++++++++')

    print(type(r_data))
    # print(type(rd_data))
    print(type(rela_data))

    # r_data = np.array(r_data)
    # print(r_data)
    print(type(r_data[1]))
    print(r_data[1][0][0])
    print(type(r_data[1][0][1]))
    print(type(r_data[1][0]))
    print(len(r_data))
    print(len(r_data[0]))
    print(r_data[0][0])
    print(len(r_data[0][0]))
    print(r_data[0][0][0])
    print(len(r_data[0][0][0]))

    print()
    # print(rd_data[0])
    # print(len(rd_data))
    # print(len(rd_data[0]))
    # print(rd_data[0][0])
    # print(len(rd_data[0][0]))
    # print(rd_data[0][0][0])
    # print(len(rd_data[0][0][0]))

    for key in rela_data:
        print(key)
    print(rela_data['inter_dict_id2inter'])
    print(rela_data['inter_dict_inter2id'])
    print(rela_data['road_dict_id2road'])
    print(rela_data['road_dict_road2id'])
    print(rela_data['inter_in_roads'])
    print(rela_data['inter_out_roads'])
    print(rela_data['road_links'])
    print(rela_data['neighbor_num'])
    print(rela_data['net_shape'])
