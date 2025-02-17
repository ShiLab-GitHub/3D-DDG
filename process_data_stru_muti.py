import torch
from transformers import T5EncoderModel, T5Tokenizer
from transformers import BertModel, BertTokenizer
from transformers import XLNetModel, XLNetTokenizer
from transformers import AlbertModel, AlbertTokenizer

from rdkit import Chem
from rdkit.Chem import MolFromSmiles

import re
import os
import pickle
import prody as pdy
import pandas as pd
import numpy as np
import networkx as nx
import requests
from tqdm.auto import tqdm
from protseqfeature import *
from utils import *
import time
import multiprocessing

pdbid2mutinfo = {}  # 使用pdb_id作为key值，chain_id res_id pdb_chain_res作为value值
# 通过pdb_id获取到蛋白质的结构信息并转换为图
pdb_id = []
chain_id = []
res_id = []
pdb_chain_res = []

for dt_name in ['skempi']:
    opts = ['train', 'test']
    for opt in opts:
        df = pd.read_csv('data/' + dt_name + '_' + opt + '.csv')
        pdb_id += list(df['pdb_id'])
        chain_id += list(df['chain_id'])
        res_id += list(df['res_id'])
        pdb_chain_res += list(df['pdb_chain_res'])

# pdbid2mutinfo = zip(df['chain_id']+'_'+str(df['res_id'])+'_'+df['pdb_chain_res'])
# pdbid2mutinfo[df['pdb_id']] = str(df['chain_id'])+'_'+str(df['res_id'])+'_'+df['pdb_chain_res']
for i, pdb_name in enumerate(pdb_id):
    pdbid2mutinfo[pdb_name] = chain_id[i] + '_' + str(res_id[i]) + '_'+pdb_chain_res[i]  # pdbid2mutinfo的索引类似于1CSE_I_LEU38GLY,value类似于I_38_1CSE_E_I


aa1_map = {aa.name_1_letter: aa for k, aa in aa3_map.items()}

aa3to1 = {aa.name_3_letter: aa.name_1_letter for k, aa in aa1_map.items()}
aa1to3 = {aa.name_1_letter: aa.name_3_letter for k, aa in aa1_map.items()}

# aa3_label = {aa: f"{i+1}" for i, aa in enumerate(sorted(aa3to1.keys()))}
aa3_label = {aa: i+1 for i, aa in enumerate(sorted(aa3to1.keys()))}  # 这个地方不能把编码设置为str类型，而是要设置为数值
aa3_label["GLU"], aa3_label["GLN"] = aa3_label["GLN"], aa3_label["GLU"]

# 获取resolution, r_value, temperature, ph四个新特征
def mutation_pdb_information(pdbid):
    resolution, r_value, temperature, ph = 0, 0, 0, 0
    for line in open(pdbid):
        pdbstr = line.strip()
        if pdbstr[0:22] == "REMARK   2 RESOLUTION.":
            try:
                resolution = float(pdbstr[26:30].strip())
            except ValueError:
                resolution = 0
        if pdbstr[0:45] == "REMARK   3   R VALUE            (WORKING SET)":
            try:
                r_value = float(pdbstr[49:54].strip())
            except ValueError:
                r_value = 0
        if pdbstr[0:23] == "REMARK 200  TEMPERATURE":
            try:
                temperature = float(pdbstr[45:48].strip())
            except ValueError:
                temperature = 0
        if pdbstr[0:14] == "REMARK 200  PH":
            ph = "".join([ch for ch in pdbstr[45:48].strip() if ch in "0123456789."])
            try:
                ph = float(ph)
            except ValueError:
                ph = 0
            break
    return resolution, r_value, temperature, ph

def mutation_distance(pdbid, chain, mutation_coordinate):
    resid_label_dis_aa = []
    resid_label_aa = []
    resid_label_distance = []
    for line in open(pdbid):
        pdbstr = line.strip()
        if pdbstr[0:4] == "ATOM" and pdbstr[13:15] == "CA" and pdbstr[21:22] != chain:
            mutation_coordinate1 = [
                float(pdbstr[29:38].strip()),
                float(pdbstr[38:46].strip()),
                float(pdbstr[46:55].strip()),
            ]
            resid_label_aa.append(pdbstr[17:20])
            resid_label_distance.append(
                np.sqrt(
                    np.square(mutation_coordinate[0] - mutation_coordinate1[0])
                    + np.square(mutation_coordinate[1] - mutation_coordinate1[1])
                    + np.square(mutation_coordinate[2] - mutation_coordinate1[2])
                )
            )
    b = list(zip(resid_label_distance, list(range(len(resid_label_distance)))))
    b.sort(key=lambda x: x[0])
    c = [x[1] for x in b]
    sequ_num = 10  # 10 sequence in total,use last 7sequence
    if len(c) >= sequ_num:
        for j in range(0, sequ_num, 1):
            if b[j][0] <= 10:
                resid_label_dis_aa.append(resid_label_aa[c[j]])
            else:
                resid_label_dis_aa.append(0)
    else:
        for j in range(0, len(c), 1):
            if b[j][0] <= 10:
                resid_label_dis_aa.append(resid_label_aa[c[j]])
            else:
                resid_label_dis_aa.append(0)
        while len(resid_label_dis_aa) < sequ_num:
            resid_label_dis_aa.append(0)
    return resid_label_dis_aa


def mutation_sequence2(struct, resid, chain, window=(-5, 5)):
    ch = chain
    resi = int("".join([c for c in resid if c in "-0123456789"]))
    wt_sel_str = f"chain {ch} and resid {resi} and name CA"
    wt_sel = struct.select(wt_sel_str)
    resid_label_aa = [0 for _ in range(window[0], window[1] + 1)]
    if not (wt_sel is None) and len(wt_sel.getResnames()) == 1:
        mt_coords = wt_sel.getCoords()[0]
        chain_ca = struct.select(f"chain {ch} and name CA")
        chain_resi = [
            (ch, ri, rn)
            for ri, rn in zip(chain_ca.getResnums(), chain_ca.getResnames())
        ]
        ri_list = [(i, e) for i, e in enumerate(chain_resi) if e[1] == resi]
        if len(ri_list) == 1:
            rindx = ri_list[0][0]
            lbl_shift = 0
            rindx_from, rindx_to = rindx + window[0], rindx + window[1] + 1
            if rindx_to > len(chain_resi):
                rindx_to = len(chain_resi)
            if window[1] - rindx > 0:
                rindx_from = 0
                lbl_shift = window[1] - rindx
            for i, e in enumerate(chain_resi[rindx_from:rindx_to]):
                resid_label_aa[lbl_shift + i] = e[2]
    else:
        return resid_label_aa, False

    return resid_label_aa, mt_coords

# 通过pdb_id获取到蛋白质的结构信息并转换为图
def process_pdb(pdb_id):

    with open('data/pdb_stru_graph.pickle', 'rb') as handle:
        pdb2path = pickle.load(handle)

    # 解析id
    id = pdb_id

    # 单独存储的主要目的是为了断电等意外之后还能保留已经运行的处理结果
    # 这块代码就是在断电重新运行的时候
    save_path = pdb2path[id]
    if os.path.exists(save_path):
        return
    pdb = id.split('_')[0]
    # gen_graph_data的第一个输入，pdb文件的地址
    # wildtypefile = './SKEMPI_pdbs/{}.pdb'.format(pdb)
    wildtypefile = './SKEMPI_all_pdbs/PDBs/{}.pdb'.format(pdb)
    # wildtypefile = './SKEMPI2_PDBs/PDBs/._{}.pdb'.format(pdb)
    # gen_graph_data的第二个输入，graph_mutinfo包含着chainid 和resid
    graph_mutinfo = []
    chainid = pdbid2mutinfo[id].split('_')[0]
    resid = pdbid2mutinfo[id].split('_')[1]
    graph_mutinfo.append('{}_{}'.format(chainid, resid))

    # 现在看第三个和第五个输入是没有用的
    # # gen_graph_data的第五个输入，if_info是蛋白质接触的两条链
    if_info = pdbid2mutinfo[id].split('_')[-2] + '_' + pdbid2mutinfo[id].split('_')[-1]
    # # gen_graph_data的第三个输入，interfacefile是interface.txt文件的地址，这个输入需要第五个输入的先决信息
    workdir = "temp"
    # # generate the interface residues
    # os.system('python gen_interface.py {} {} {} > {}/pymol.log'.format(wildtypefile, if_info, workdir, workdir))
    interfacefile = '{}/interface.txt'.format(workdir)

    # cutoff的第四个输入，GeoPPI中固定设置为3
    cutoff = 3

    # 每个PDB的图表征单独先存储
    g = gen_graph_data(wildtypefile, graph_mutinfo, interfacefile, cutoff, if_info)
    # save_path = pdb2path[id]
    with open(save_path, 'wb') as f:
        pickle.dump(g, f)


def build_graph(lines, interface_res, mutinfo, cutoff=3, max_dis=12, pdb_path=''):
    atomnames = ['C', 'N', 'O', 'S']
    residues = ['ARG', 'MET', 'VAL', 'ASN', 'PRO', 'THR', 'PHE', 'ASP', 'ILE', \
                'ALA', 'GLY', 'GLU', 'LEU', 'SER', 'LYS', 'TYR', 'CYS', 'HIS', 'GLN', 'TRP']
    res_code = ['R', 'M', 'V', 'N', 'P', 'T', 'F', 'D', 'I', \
                'A', 'G', 'E', 'L', 'S', 'K', 'Y', 'C', 'H', 'Q', 'W']
    res2code = {x: idxx for x, idxx in zip(residues, res_code)}

    atomdict = {x: i for i, x in enumerate(atomnames)}
    resdict = {x: i for i, x in enumerate(residues)}
    V_atom = len(atomnames)
    V_res = len(residues)

    # build chain2id
    chain2id = []
    interface_coordinates = []
    line_list = []
    mutant_coords = []
    # CA C N骨干原子的坐标记录
    main_coords = {}
    main_coords1 = []
    all_coords = []
    atom_st = 0
    main_coords1_index = 0
    ###这一部分是不是不需要啊
    for line in lines:
        if line[0:4] == 'ATOM':
            atomname = line[12:16].strip()
            elemname = list(filter(lambda x: x.isalpha(), atomname))[0]
            resname = line[16:21].strip()
            chainid = line[21]
            res_idx = line[22:28].strip()
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            if elemname not in atomdict:
                continue

            coords = torch.tensor([x, y, z])
            # atomid = atomdict[elemname]
            if resname not in resdict:
                resname = resname[1:]
            if resname not in resdict:
                continue

            if chainid not in chain2id:
                chain2id.append(chainid)

            line_token = '{}_{}_{}_{}'.format(atomname, resname, chainid, res_idx)
            if line_token not in line_list:
                line_list.append(line_token)
            else:
                continue
            if atomname == 'CA' or atomname == 'C' or atomname == 'N':
                main_coords1.append(coords)
                main_coords[atom_st] = main_coords1_index
                main_coords1_index = main_coords1_index + 1
            all_coords.append(coords)
            atom_st = atom_st+1

    # 搜集的骨干原子的个数
    main_coords_length = len(main_coords1)
    all_coords_length = len(all_coords)
    chain2id = {x: i for i, x in enumerate(chain2id)}
    ###这一部分是不是不需要啊

    n_features = V_atom + V_res + 12
    line_list = []
    atoms = []
    ATOM_ID = []
    res_index_set = {}
    atom_st = 0
    for line in lines:
        if line[0:4] == 'ATOM':
            # print('看看新的pdb文件的atom内容', line, line[4:12].strip(), line[12:16].strip(), line[21], line[22:28].strip())
            features = [0] * n_features
            # ATOM_ID.append(int(line[4:12].strip()))  # 为了后面构建索引
            atomname = line[12:16].strip()
            elemname = list(filter(lambda x: x.isalpha(), atomname))[0]
            resname = line[16:21].strip()
            chainid = line[21]
            res_idx = line[22:28].strip()
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            if elemname not in atomdict:
                continue

            coords = torch.tensor([x, y, z])
            # print("看看coords", coords)
            atomid = atomdict[elemname]
            if resname not in resdict:
                resname = resname[1:]
            if resname not in resdict:
                continue
            line_token = '{}_{}_{}_{}'.format(atomname, resname, chainid, res_idx)
            if line_token not in line_list:
                line_list.append(line_token)
            else:
                continue

            resid = resdict[resname]
            features[atomid] = 1
            features[V_atom + resid] = 1

            cr_token = '{}_{}'.format(chainid, res_idx)
            float_cd = [float(x) for x in coords]
            # print("看看float_cd",float_cd)


            # 这个atoms才是最后的特征吧
            # atoms.append(features)
            ATOM_ID.append(int(line[4:12].strip()))  # 为了后面构建索引

            struct = pdy.parsePDB(pdb_path, header=False)
            # 第25个特征
            if mutinfo is not None:
                for inforrr in mutinfo:
                                mut_chainid = inforrr.split('_')[0]
                                if chainid == mut_chainid:
                                    features[V_atom + V_res] = 1
            # 第26个特征
            features[V_atom + V_res + 1] = chain2id[chainid]
            # 第27个特征
            if cr_token not in res_index_set:
               res_index_set[cr_token] = len(res_index_set) + 1
            features[V_atom + V_res + 2] = res_index_set[cr_token]
            # 第28个特征
            if atomname == 'CA':
               features[V_atom + V_res + 3] = res_index_set[cr_token]
            # 第29-31个特征
            features[V_atom + V_res + 4:V_atom + V_res + 7] = float_cd
            # 第32个特征
            if mutinfo is not None and cr_token in mutinfo:
                features[V_atom + V_res + 7] = 1
            # 第33-36个特征
            (reso, r_value, temp, ph) = mutation_pdb_information(pdb_path)
            features[V_atom + V_res + 8] = reso
            features[V_atom + V_res + 9] = r_value
            features[V_atom + V_res + 10] = temp
            features[V_atom + V_res + 11] = ph

            # 第37-38个特征
            # if mutinfo is not None:
            #     for inforrr in mutinfo:
            #         mutation_chain = inforrr.split('_')[0]
            #         mutation_resid = inforrr.split('_')[1]
            #         (resid_label_aa, mutation_coordinate) = mutation_sequence2(
            #             struct, str(mutation_resid), mutation_chain
            #         )
            #         if isinstance(mutation_coordinate, bool):
            #             features[V_atom + V_res + 12] = 0
            #             features[V_atom + V_res + 13] = 0
            #         else:
            #             label_aa_distance = mutation_distance(pdb_path, mutation_chain, mutation_coordinate)
            #             for i in resid_label_aa:
            #                 # print(aa3_label.get(i, 0))
            #                 features[V_atom + V_res + 12] = aa3_label.get(i, 0)
            #             for i in label_aa_distance:
            #                 # print(aa3_label.get(i, 0))
            #                 features[V_atom + V_res + 13] = aa3_label.get(i, 0)

            # # 第39-40个特征phi,psi
            #
            # if atomname == 'CA':
            #    the_index = main_coords[atom_st]
            #    if main_coords_length-3 >= the_index >= 2:
            #        phi = calculate_dihedral(main_coords1[the_index-2], main_coords1[the_index-1], main_coords1[the_index], main_coords1[the_index+1])
            #        psi = calculate_dihedral(main_coords1[the_index-1], main_coords1[the_index], main_coords1[the_index+1], main_coords1[the_index+2])
            #        # if not(phi == 0 or psi == 0):
            #        #     min_p = min(phi, psi)
            #        #     phi = phi / min_p
            #        #     psi = psi / min_p
            #        # print(phi, psi)
            #        # sum_p = psi+phi
            #        # phi = phi / sum_p
            #        # psi = psi / sum_p
            #    else:
            #        phi = 0
            #        psi = 0
            #
            #    features[V_atom + V_res + 14] = phi
            #    features[V_atom + V_res + 15] = psi
            # else:
            #     if all_coords_length - 3 >= atom_st >= 2:
            #         phi = calculate_dihedral(all_coords[atom_st - 2], all_coords[atom_st - 1], all_coords[atom_st],
            #                                  all_coords[atom_st + 1])
            #         psi = calculate_dihedral(all_coords[atom_st - 1], all_coords[atom_st], all_coords[atom_st + 1],
            #                                  all_coords[atom_st + 2])
            #     else:
            #         phi = 0
            #         psi = 0
            #     features[V_atom + V_res + 14] = phi
            #     features[V_atom + V_res + 15] = psi
            # atom_st = atom_st + 1

            atoms.append(features)

    # 是不是这个地方啊
    # if len(atoms) < 5:
    #     return None
    if len(atoms) < 5:
        print("存在原子长度小于5的残基长度为：{}".format(len(atoms)))
    atoms = torch.tensor(atoms, dtype=torch.float)
    N = atoms.size(0)
    atoms_type = torch.argmax(atoms[:, :4], 1)
    atoms_type = atoms_type.unsqueeze(1).repeat(1, N)
    edge_type = atoms_type * 4 + atoms_type.t()

    pos = atoms[:, -4:-1]  # N,3
    row = pos[:, None, :].repeat(1, N, 1)
    col = pos[None, :, :].repeat(N, 1, 1)
    direction = row - col
    del row, col
    distance = torch.sqrt(torch.sum(direction ** 2, 2)) + 1e-10
    distance1 = (1.0 / distance) * (distance < float(cutoff)).float()
    del distance
    diag = torch.diag(torch.ones(N))
    dist = diag + (1 - diag) * distance1
    del distance1, diag
    flag = (dist > 0).float()
    direction = direction * flag.unsqueeze(2)
    del direction, dist
    edge_sparse = torch.nonzero(flag)  # K,2
    edge_attr_sp = edge_type[edge_sparse[:, 0], edge_sparse[:, 1]]  # K,4

    # 制造edges_index
    edges = []

    # for atom in range(len(ATOM_ID)-1):
    #     edges.append([ATOM_ID[atom], ATOM_ID[atom+1]])  # 获取键（bond）两端的原子的索引
    for atom in range(N-1):
        edges.append([atom, atom+1])  # 获取键（bond）两端的原子的索引
    g = nx.Graph(edges).to_directed()  # 构建图
    edge_index = []
    for e1, e2 in g.edges:
        # 之前做edges_index的时候好像弄错了，因为
        # edge_index.append([e2, e1])  # 构建的图的edge索引，一般dege_index就是一个两列的向量，第一列的向量的第n位指向第二列的第n位
        edge_index.append([e1, e2])  # 构建的图的edge索引，一般dege_index就是一个两列的向量，第一列的向量的第n位指向第二列的第n位
        # 在这个构图的过程中使用的是顺序的构图方式，也就是上一段原子bond的开头指向下一段原子的开头，以这种方式构建图。
        # 构成图的连接方式其实就是一个键（bond）两端的原子进行对应连接

    print("atoms,ATOM_ID, edges,edges_index的维度", len(atoms), len(ATOM_ID),len(edges), len(edge_index), )

    # 如果使用SKEMPI_all_pdbs下面的pdb文件的话，必须保证len(edge_index)>=ATOM_ID保存的最大值
    return N, atoms, edge_index


def gen_graph_data(pdbfile, mutinfo, interfile,  cutoff, if_info=None):
    max_dis = 12
    # pdb_path是pdb文件的路径
    pdb_path = pdbfile
    # 这里做一个判断是否pdbfile是存在于SKEMPI_pdbs文件中的
    # if os.path.exists(pdbfile):
    #     pdbfile = open(pdbfile,encoding='UTF-8')
    #     # pdbfile = open(pdbfile,encoding='ISO-8859-1')
    #     lines = pdbfile.read().splitlines()
    #     # chainid = [x.split('_')[0] for x in mutinfo]
    #     # interface_res = read_inter_result(interfile,if_info, chainid)
    #     # if len(interface_res)==0: print('Warning: We do not find any interface residues between the two parts: {}. Please double check your inputs. Thank you!'.format(if_info))
    #     interface_res = None
    #     sample = build_graph(lines, interface_res,mutinfo, cutoff,max_dis)
    #     return sample
    # else:
    #     return "chuwenti"

    pdbfile = open(pdbfile,encoding='UTF-8')
    # pdbfile = open(pdbfile,encoding='ISO-8859-1')
    lines = pdbfile.read().splitlines()
    # chainid = [x.split('_')[0] for x in mutinfo]
    # interface_res = read_inter_result(interfile,if_info, chainid)
    # if len(interface_res)==0: print('Warning: We do not find any interface residues between the two parts: {}. Please double check your inputs. Thank you!'.format(if_info))
    interface_res = None
    sample = build_graph(lines, interface_res,mutinfo, cutoff,max_dis, pdb_path)
    return sample

if __name__ == "__main__":

    pdb_id = []
    for dt_name in ['skempi']:
        opts = ['train', 'test']
        for opt in opts:
            df = pd.read_csv('data/' + dt_name + '_' + opt + '.csv')
            pdb_id += list(df['pdb_id'])
    pdb_id = set(pdb_id)
    pdb_stru_graph = {}

    # 存储单个图表征的文件所在路径
    pdb_graph_dir = './data/pdb_graph'
    pdb2path = {v: os.path.join(pdb_graph_dir, '{}.pickle'.format(v)) for i, v in
                enumerate(pdb_id)}  # 字典，key为pdb_id、value为该pdb生成的图表征的保存pickle文件路径
    with open('data/pdb_stru_graph.pickle', 'wb') as handle:
        pickle.dump(pdb2path, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #多进程
    start = time.time()
    print("父进程：{}".format(os.getpid()))

    # with open('data/pdb_stru_graph.pickle', 'rb') as handle:
    #     pdb2path = pickle.load(handle)

    pool = (multiprocessing.Pool(processes=os.cpu_count()))
    pool.map(process_pdb,pdb_id)
    pool.close()
    pool.join()
    end = time.time()
    print((end - start))