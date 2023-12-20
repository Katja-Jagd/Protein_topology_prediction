import torch
from typing import List, Union

def label_list_to_topology(labels: Union[List[int], torch.Tensor]) -> List[torch.Tensor]:
    """
    Converts a list of per-position labels to a topology representation.
    This maps every sequence to list of where each new symbol start (the topology), e.g. AAABBBBCCC -> [(0,A),(3, B)(7,C)]

    Parameters
    ----------
    labels : list or torch.Tensor of ints
        List of labels.

    Returns
    -------
    list of torch.Tensor
        List of tensors that represents the topology.
    """

    if isinstance(labels, list):
        labels = torch.LongTensor(labels)

    if isinstance(labels, torch.Tensor):
        zero_tensor = torch.LongTensor([0])
        if labels.is_cuda:
            zero_tensor = zero_tensor.cuda()

        unique, count = torch.unique_consecutive(labels, return_counts=True)
        top_list = [torch.cat((zero_tensor, labels[0:1]))]
        prev_count = 0
        i = 0
        for _ in unique.split(1):
            if i == 0:
                i += 1
                continue
            prev_count += count[i - 1]
            top_list.append(torch.cat((prev_count.view(1), unique[i].view(1))))
            i += 1
        return top_list


def is_topologies_equal(topology_a, topology_b, minimum_seqment_overlap=5):
    """
    Checks whether two topologies are equal.
    E.g. [(0,A),(3, B)(7,C)]  is the same as [(0,A),(4, B)(7,C)]
    But not the same as [(0,A),(3, C)(7,B)]

    Parameters
    ----------
    topology_a : list of torch.Tensor
        First topology. See label_list_to_topology.
    topology_b : list of torch.Tensor
        Second topology. See label_list_to_topology.
    minimum_seqment_overlap : int
        Minimum overlap between two segments to be considered equal.

    Returns
    -------
    bool
        True if topologies are equal, False otherwise.
    """

    if isinstance(topology_a[0], torch.Tensor):
        topology_a = list([a.cpu().numpy() for a in topology_a])
    if isinstance(topology_b[0], torch.Tensor):
        topology_b = list([b.cpu().numpy() for b in topology_b])
    if len(topology_a) != len(topology_b):
        return False
    for idx, (_position_a, label_a) in enumerate(topology_a):
        if label_a != topology_b[idx][1]:
            if (label_a in (1,2) and topology_b[idx][1] in (1,2)): # assume O == P
                continue
            else:
                return False
        if label_a in (3, 4, 5):
            overlap_segment_start = max(topology_a[idx][0], topology_b[idx][0])
            overlap_segment_end = min(topology_a[idx + 1][0], topology_b[idx + 1][0])
            if label_a == 5:
                # Set minimum segment overlap to 3 for Beta regions
                minimum_seqment_overlap = 3
            if overlap_segment_end - overlap_segment_start < minimum_seqment_overlap:
                return False
    return True


def calculate_acc(correct, total):
    total = total.float()
    correct = correct.float()
    if total == 0.0:
        return 1
    return correct / total


def type_from_labels(label):
    """
    Function that determines the protein type from labels

    Dimension of each label:
    (len_of_longenst_protein_in_batch)

    # Residue class
    0 = inside cell/cytosol (I)
    1 = Outside cell/lumen of ER/Golgi/lysosomes (O)
    2 = beta membrane (B)
    3 = signal peptide (S)
    4 = alpha membrane (M)
    5 = periplasm (P)

    B in the label sequence -> beta
    I only -> globular
    Both S and M -> SP + alpha(TM)
    M -> alpha(TM)
    S -> signal peptide

    # Protein type class
    0 = TM
    1 = SP + TM
    2 = SP
    3 = GLOBULAR
    4 = BETA
    """

    if 2 in label:
        ptype = 4

    elif all(element == 0 for element in label):
        ptype = 3

    elif 3 in label and 4 in label:
        ptype = 1

    elif 3 in label:
       ptype = 2

    elif 4 in label:
        ptype = 0

    elif all(x == 0 or x == -1 for x in label):
        ptype = 3

    else:
        ptype = None

    return ptype


def bio_acc(output, target):
    """
    Functions that calculates a biological accuracy based on both
    topology and protein type predictions
    """

    confusion_matrix = torch.zeros((6,6),dtype = torch.int64)

    for i in range(len(output)):
        for j in range(len(output[i])):

            # Get shortened outputs and targets, removing -1 padding
            if (target[i][j] == -1).any():
                tmp_output = output[i][j][0:(target[i][j] == -1).nonzero()[0].item()]
                tmp_target = target[i][j][0:(target[i][j] == -1).nonzero()[0].item()]
            else:
                tmp_output = output[i][j]
                tmp_target = target[i][j]

            # Get topology
            output_topology = label_list_to_topology(tmp_output)
            target_topology = label_list_to_topology(tmp_target)

            # Get protein type
            output_type = type_from_labels(tmp_output)
            target_type = type_from_labels(tmp_target)


            # Check if topologies match
            prediction_topology_match = is_topologies_equal(output_topology, target_topology, 5)

            if target_type == output_type:
                # if we guessed the type right for SP+GLOB or GLOB,
                # count the topology as correct
                if target_type == 2 or target_type == 3 or prediction_topology_match:
                    confusion_matrix[target_type][5] += 1
                else:
                    confusion_matrix[target_type][output_type] += 1

            else:
                confusion_matrix[target_type][output_type] += 1

    # Calculate individual class accuracy for protein type prediction
    tm_type_acc = float(calculate_acc(confusion_matrix[0][0] + confusion_matrix[0][5], confusion_matrix[0].sum()))
    tm_sp_type_acc = float(calculate_acc(confusion_matrix[1][1] + confusion_matrix[1][5], confusion_matrix[1].sum()))
    sp_type_acc = float(calculate_acc(confusion_matrix[2][2] + confusion_matrix[2][5], confusion_matrix[2].sum()))
    glob_type_acc = float(calculate_acc(confusion_matrix[3][3] + confusion_matrix[3][5], confusion_matrix[3].sum()))
    beta_type_acc = float(calculate_acc(confusion_matrix[4][4] + confusion_matrix[4][5], confusion_matrix[4].sum()))

    # Calculate individual class accuracy for protein topology prediction
    tm_accuracy = float(calculate_acc(confusion_matrix[0][5], confusion_matrix[0].sum()))
    sptm_accuracy = float(calculate_acc(confusion_matrix[1][5], confusion_matrix[1].sum()))
    sp_accuracy = float(calculate_acc(confusion_matrix[2][5], confusion_matrix[2].sum()))
    glob_accuracy = float(calculate_acc(confusion_matrix[3][5], confusion_matrix[3].sum()))
    beta_accuracy = float(calculate_acc(confusion_matrix[4][5], confusion_matrix[4].sum()))

    # Calculate average accuracy for protein type prediction
    type_accuracy = (tm_type_acc + tm_sp_type_acc + sp_type_acc + glob_type_acc + beta_type_acc) / 5

    # Calculate average accuracy for protein topology prediction
    topology_accuracy = (tm_accuracy + sptm_accuracy + sp_accuracy + glob_accuracy + beta_accuracy) / 5

    # Combined accuracy score for topology and type prediction
    total_accuracy = (type_accuracy + topology_accuracy) / 2

    return confusion_matrix, total_accuracy
