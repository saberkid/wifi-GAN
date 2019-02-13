import numpy as np
import os, sys
import warnings
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import deque
import scipy.io as sio
import matplotlib.widgets as widgets
import struct
import scipy
from scipy import signal
from datetime import datetime, timedelta
from pytz import timezone
import pytz
from dateutil import tz
from scipy.stats import norm

# %matplotlib inline

thresholds_dict = dict({'eyadogupinon': 25,
                        '9897D173087F': 30,
                        'aponaqatebin': 30,  # 25 till 2018-06-13
                        '9897D1483F83': 40,
                        '84AA9C0BAFEF': 15,
                        'edewuwamosed': 25,
                        'ezuzoduxuhoc': 25,
                        '84AA9C0BB0D5': 25,
                        'aqumisifijap': 15,
                        'idedigizovuc': 25,
                        'ezanoxepadur': 25,
                        '9897D14840D7': 20,  # since April 09, 2018
                        'ukequkutapux': 15,
                        'orowuhadikov': 15,
                        '84AA9C0BB1B1': 25,
                        'azepiwubirun': 25,
                        '84AA9C0BB1BB': 25,
                        '345760DFD2D3': 15,
                        'oxefafebexup': 25,
                        'E04136714351': 30,
                        'E041367DAAED': 25,
                        '34576091ED38': 25,
                        '84AA9C4BFFF5': 25,
                        'B046FC8078D4': 15,
                        '84AA9C27BC89': 25,
                        'uginutesukig': 25,
                        'E041367DF63B': 35,
                        '84AA9C2F4152': 25,
                        'E04136BE687E': 20,
                        '345760917054': 25,
                        '9897D10C7AF0': 25,
                        'agutotopeyoc': 25,
                        'esuwehaqefic': 25,
                        'aquyecusutew': 25,
                        'odazezicayin': 25
                        })

stream_codes_dict = dict({'0001': 0, '0010': 4, '0100': 8, '1000': 12,
                          '0002': 1, '0020': 5, '0200': 9, '2000': 13,
                          '0004': 2, '0040': 6, '0400': 10, '4000': 14,
                          '0008': 3, '0080': 7, '0800': 11, '8000': 15
                          })


def map_stream(stream_code):
    return stream_codes_dict[stream_code]


def _get_local_path():
    local_path = 'data/raw/002686F0C609/Drops'
    # local_path = 'data/raw/002686F0D3B3/Spike_Armed'
    print('local path = {}'.format(local_path))

    if not local_path:
        raise Exception('AERIAL_CSI_DATA is not set!')
    # add the trailing / if not already present
    if not local_path.endswith('/'):
        local_path += '/'
    # verify the path is valid
    if not os.path.isdir(local_path):
        raise Exception('AERIAL_CSI_DATA is not a folder! ("{}")'.format(local_path))
    return local_path


# This function returns a list of the full filepaths from a speficied filenames
def filepaths_from_files(files):
    local_path = _get_local_path()
    file_paths = []
    # validate the files are present
    for f in files:
        file_path = local_path + f
        if not os.path.isfile(file_path):
            raise Exception('File "{}" was not found!'.format(file_path))
        file_paths.append(file_path)
    return file_paths


# This function reads a csi filename string and returns the csi specifications
def compile_file_str(file_str, stream_loc):
    file_splitted = file_str.split('_')

    device_name0 = file_splitted[0].split('_')[0]
    device_name = device_name0.split('/')[-1]

    stream_code = file_splitted[stream_loc]
    stream = map_stream(stream_code)
    threshold = thresholds_dict[device_name]
    return device_name, stream, threshold


# This function reads a .csi file and outputs a packets list with device and
# version information.
def csi_to_packets(csiFile):
    # print('csiFile = {}'.format(csiFile))
    index = 0
    packets = []
    packetCount = 0

    # ----------------------------------------------------------------------------------------------------------------
    def stream_is_enabled(nTx, nRx, streamMask, iTx, iRx):
        maxStream = nTx * nRx
        streamIndex = (nTx * iTx) + iRx

        if streamIndex >= maxStream or streamIndex >= 16:
            print('Error, invalid stream index! (%d)' % streamIndex)
            return False

        return streamMask & (1 << streamIndex)

    # ----------------------------------------------------------------------------------------------------------------
    def get_packet_csi_matrix(matrixBuf, nsc, ntx, nrx, streamMask, bitsPerSymbol):

        def bit_convert(num, bitsPerSymbol):
            if num & (1 << (bitsPerSymbol - 1)):
                num -= (1 << bitsPerSymbol)
            return num

        def read_two(buffer, index):
            try:
                return struct.unpack('=H', buffer[index: index + 2])[0] & 0xFFFF
            except:
                pass

        csiMatrix = np.zeros(shape=(nsc, ntx, nrx), dtype=complex, order='C')

        bits_left = 16
        bit_mask = (1 << bitsPerSymbol) - 1
        index = 0
        current_data = read_two(matrixBuf, index)
        index += 2

        for subcarrier in range(0, nsc):
            for transmitter in range(0, ntx):
                for receiver in range(0, nrx):
                    if stream_is_enabled(ntx, nrx, streamMask, transmitter, receiver):

                        if bits_left - bitsPerSymbol < 0:
                            current_data += read_two(matrixBuf, index) << bits_left
                            index += 2
                            bits_left += 16

                        imag = bit_convert(current_data & bit_mask, bitsPerSymbol)
                        bits_left -= bitsPerSymbol
                        current_data >>= bitsPerSymbol

                        if bits_left - bitsPerSymbol < 0:
                            current_data += read_two(matrixBuf, index) << bits_left
                            index += 2
                            bits_left += 16

                        real = bit_convert(current_data & bit_mask, bitsPerSymbol)
                        bits_left -= bitsPerSymbol
                        current_data >>= bitsPerSymbol

                        csiMatrix[subcarrier][transmitter][receiver] = real + 1j * imag

        return csiMatrix

    # The different header format based on the version
    PACKET_HEADER_FORMAT_V00_09_XX = '= Q 2H h H 7B 4h B 6B'
    PACKET_HEADER_SIZE_V00_09_XX = struct.calcsize(PACKET_HEADER_FORMAT_V00_09_XX)

    PACKET_HEADER_FORMAT_V00_10_XX = '= I Q 2H h H 7B 4h B 6B'
    PACKET_HEADER_SIZE_V00_10_XX = struct.calcsize(PACKET_HEADER_FORMAT_V00_10_XX)

    PACKET_HEADER_FORMAT_V01 = '= I Q 2H h H 7B 4h B 6B'
    PACKET_HEADER_SIZE_V01 = struct.calcsize(PACKET_HEADER_FORMAT_V01)

    PACKET_HEADER_FORMAT_V03 = '= I Q 2H h B H 3B H 3B 4h B 6B'
    PACKET_HEADER_SIZE_V03 = struct.calcsize(PACKET_HEADER_FORMAT_V03)

    with open(csiFile, 'rb') as file:
        buffer = file.read()
        # Read the device ID
        device = struct.unpack('=3B', buffer[index: index + 3])
        index += 3
        # print("Unpacking %s for device %s" % (csiFile, device))

        # Check if the version identifier is present (after v00.10.04)
        agentVersion = struct.unpack('=10B', buffer[index: index + 10])
        agentVersion = [chr(val) for val in agentVersion]
        agentVersion = ''.join(agentVersion)
        if agentVersion[0] == 'v' and agentVersion[1] == '.':
            # Agent version found!
            index += 10
        else:
            print("Agent version not found... Using v.00.09.xx ")
            agentVersion = "v.00.09.xx"

        while True:
            try:
                packetSize = struct.unpack('=H', buffer[index: index + 2])[0] & 0xFFFF
            except:
                # print ("Processed %d packets" % packetCount)
                break

            index += 2
            if packetSize > 0:
                if agentVersion[0: 7] == "v.00.09":
                    packetCount += 1
                    header = struct.unpack(PACKET_HEADER_FORMAT_V00_09_XX,
                                           buffer[index:index + PACKET_HEADER_SIZE_V00_09_XX])
                    index += PACKET_HEADER_SIZE_V00_09_XX

                    output = {
                        'timestamp': header[0],
                        'csi_length': header[1],
                        'channel': header[2],
                        'noise': header[3],
                        'tx_streams': header[4],
                        'error_info': header[5],
                        'rate': header[6],
                        'bandwidth': header[7],
                        'sc': header[8],
                        'nrx': header[9],
                        'ntx': header[10],
                        'ng': header[11],
                        'rssi': header[12:16],
                        'bits_per_symbol': header[16],
                        'clientMAC': header[17:23]
                    }

                elif agentVersion[0: 7] == "v.00.10":
                    packetCount += 1
                    header = struct.unpack(PACKET_HEADER_FORMAT_V00_10_XX,
                                           buffer[index:index + PACKET_HEADER_SIZE_V00_10_XX])
                    index += PACKET_HEADER_SIZE_V00_10_XX

                    output = {
                        'packet_id': header[0],
                        'timestamp': header[1],
                        'csi_length': header[2],
                        'channel': header[3],
                        'noise': header[4],
                        'tx_streams': header[5],
                        'error_info': header[6],
                        'rate': header[7],
                        'bandwidth': header[8],
                        'sc': header[9],
                        'nrx': header[10],
                        'ntx': header[11],
                        'ng': header[12],
                        'rssi': header[13:17],
                        'bits_per_symbol': header[17],
                        'clientMAC': header[18:24]
                    }
                elif (agentVersion[0:4] == "v.01" or
                      agentVersion[0:4] == "v.02" or
                      agentVersion[0:7] == "v.03.00"):
                    packetCount += 1
                    header = struct.unpack(PACKET_HEADER_FORMAT_V01,
                                           buffer[index:
                                                  (index +
                                                   PACKET_HEADER_SIZE_V01)])
                    index += PACKET_HEADER_SIZE_V01

                    output = {
                        'packet_id': header[0],
                        'timestamp': header[1],
                        'csi_length': header[2],
                        'channel': header[3],
                        'noise': header[4],
                        'tx_streams': header[5],
                        'error_info': header[6],
                        'rate': header[7],
                        'bandwidth': header[8],
                        'sc': header[9],
                        'nrx': header[10],
                        'ntx': header[11],
                        'ng': header[12],
                        'rssi': header[13:17],
                        'bits_per_symbol': header[17],
                        'clientMAC': ':'.join([str(hex(mac_i)[2:]) for mac_i in header[18:24]])
                    }
                elif agentVersion[0:4] == "v.03":
                    packetCount += 1
                    header = struct.unpack(PACKET_HEADER_FORMAT_V03,
                                           buffer[index:
                                                  (index +
                                                   PACKET_HEADER_SIZE_V03)])
                    index += PACKET_HEADER_SIZE_V03

                    output = {
                        'packet_id': header[0],
                        'timestamp': header[1],
                        'csi_length': header[2],
                        'channel': header[3],
                        'noise': header[4],
                        'mode': header[5],
                        'tx_streams': header[6],
                        'error_info': header[7],
                        'rate': header[8],
                        'bandwidth': header[9],
                        'sc': header[10],
                        'nrx': header[11],
                        'ntx': header[12],
                        'ng': header[13],
                        'rssi': header[14:18],
                        'bits_per_symbol': header[18],
                        'clientMAC': ':'.join([str(hex(mac_i)[2:]) for mac_i in header[19:25]])
                    }
                else:
                    raise Exception

                csiLength = output['csi_length']

                output['csi'] = get_packet_csi_matrix(buffer[index:index + csiLength], output['sc'], output['ntx'],
                                                      output['nrx'], output['tx_streams'], output['bits_per_symbol'])
                index += csiLength

                packets.append(output)

            else:
                break

    # print ("Read %d packets!" % packetCount)
    return packets, device, agentVersion


# This function gets a list of packets as input and outputs a data array with
# some information about the packets provided.
def packets_to_data_arrays(packets):
    '''
    data, pkt1_subs, pkt1_ntx, pkt1_nrx, nb_pkts = scale_parse_agent_all(act_samples)
    '''
    CSI_MAX_NRX = 4
    CSI_MAX_NTX = 4
    CSI_MAX_NSC = 122

    nb_pkts = len(packets)

    packet_nsc = max(packet['sc'] for packet in packets)

    n = 0

    data = {
        'Rssi': np.zeros((CSI_MAX_NRX, nb_pkts)),
        'Rate': np.zeros((1, nb_pkts)),
        'Noise': np.zeros((1, nb_pkts)),
        'Nrx': np.zeros((1, nb_pkts)),
        'Ntx': np.zeros((1, nb_pkts)),
        'Timestamp': np.zeros((1, nb_pkts)),
        'Subcarriers': np.zeros((1, nb_pkts)),
        'Bandwidth': np.zeros((1, nb_pkts)),
        'Error': np.zeros((1, nb_pkts)),
        # 'Client': char(' ' * ones(32, nb_pkts)),
        'Is_ac': np.zeros((1, nb_pkts)),
        'Is_mu': np.zeros((1, nb_pkts)),
        'Is_grp': np.zeros((1, nb_pkts)),
        'Ng': np.zeros((1, nb_pkts)),
        'Codebook': np.zeros((1, nb_pkts)),
        'Is_os': np.zeros((1, nb_pkts)),
        'Token': np.zeros((1, nb_pkts)),
        'SNR': np.zeros((CSI_MAX_NRX, nb_pkts)),
        'Hw_noise': np.zeros((1, nb_pkts)),
        'H': np.zeros((packet_nsc, nb_pkts, CSI_MAX_NRX * CSI_MAX_NTX), dtype=np.complex128),
        'Channel': np.zeros((1, nb_pkts)),
        'Packet_id': np.zeros((1, nb_pkts))}

    for packet in packets:
        data['Rssi'][:, n] = packet['rssi']
        data['Rate'][0, n] = packet['rate']
        data['Noise'][0, n] = packet['noise']
        data['Nrx'][0, n] = packet['nrx']
        data['Ntx'][0, n] = packet['ntx']
        data['Timestamp'][0, n] = packet['timestamp']
        data['Subcarriers'][0, n] = packet['sc']
        data['Bandwidth'][0, n] = packet['bandwidth']
        data['Error'][0, n] = packet['error_info']
        data['Channel'][0, n] = packet['channel']
        if "Packet_id" in packet.keys():
           data['Packet_id'][0, n] = packet['packet_id']
           data['Noise'][0, n] = packet['noise']
        if hasattr(packet, 'is_ac'):
            data['Is_ac'][0, n] = packet['is_ac']
        if hasattr(packet, 'is_mu'):
            data['Is_mu'][0, n] = packet['is_mu']
        if hasattr(packet, 'is_grp'):
            data['Is_grp'][0, n] = packet['is_grp']
        if hasattr(packet, 'codebook'):
            data['Codebook'][0, n] = packet['codebook']
        if hasattr(packet, 'snrs'):
            data['SNR'][:, n] = packet['snrs']
        if hasattr(packet, 'token'):
            data['Token'][0, n] = packet['token']
        if hasattr(packet, 'hw_noise'):
            data['Hw_noise'][:, n] = packet['hw_noise']
        data['Ng'][0, n] = packet['ng']
        csi_matrix = packet['csi']

        for i_tx in range(packet['ntx']):
            for i_rx in range(packet['nrx']):
                stream = (packet['nrx'] * i_tx) + i_rx
                # This 3D matrix contains all CSI matrices for the different streams
                data['H'][0:packet['sc'], n, stream] = csi_matrix[:, i_tx, i_rx]
        n += 1
    return data, packet_nsc, CSI_MAX_NTX, CSI_MAX_NRX, nb_pkts


# This function reads a .csi file and returns the csi data
def csi_to_data(csi_file):
    packets, device, version = csi_to_packets(csi_file)
    return packets_to_data_arrays(packets)


# This function get one or 3 csi paths and returns the raw CSI data,  Rssi and Ntx
def get_data_Rssi_Ntx_from_files(csi_paths, files, stream_loc=4):
    # stream_loc = loc
    l = len(csi_paths)
    if l == 1:
        i = 0
        device_name, stream, threshold = compile_file_str(files[0], stream_loc)
        print('device_name = {}, threshold = {}, stream = {}'.format(device_name, threshold, stream))
        csi_path = csi_paths[i]
        data01, pkt1_subs, pkt1_ntx, pkt1_nrx, nb_pkts = csi_to_data(csi_path)
        Ntx01 = data01['Ntx'][0]
        H01 = data01['H']
        Rssi01 = data01['Rssi'].T

        str1 = ('|H1| = ({}, {}, {})'.format(len(H01), len(H01[0]), len(H01[0][0])))
        str2 = ''
        str3 = ''
        data0 = H01
        Rssi = Rssi01
        Ntx = Ntx01

    elif l == 2:
        i = 0
        device_name, stream, threshold = compile_file_str(files[1], stream_loc)
        print('device_name = {}, threshold = {}, stream = {}'.format(device_name, threshold, stream))
        csi_path = csi_paths[i]
        data01, pkt1_subs, pkt1_ntx01, pkt1_nrx, nb_pkts = csi_to_data(csi_path)
        Ntx01 = data01['Ntx'][0]
        H01 = data01['H']
        Rssi01 = data01['Rssi'].T

        csi_path = csi_paths[i + 1]
        data02, pkt1_subs, pkt1_ntx02, pkt1_nrx, nb_pkts = csi_to_data(csi_path)
        Ntx02 = data02['Ntx'][0]
        H02 = data02['H']
        Rssi02 = data02['Rssi'].T

        str1 = ('|H1| = ({}, {}, {})'.format(len(H01), len(H01[0]), len(H01[0][0])))
        str2 = ('|H2| = ({}, {}, {})'.format(len(H02), len(H02[0]), len(H02[0][0])))
        str3 = ''
        # print(len(H02), len(H02[0]), len(H02[0][0]))

        data0 = np.concatenate((H01, H02), axis=1)
        Rssi = np.concatenate((Rssi01, Rssi02), axis=0)
        Ntx = np.hstack([Ntx01, Ntx02])

    elif l == 3:
        i = 0
        device_name, stream, threshold = compile_file_str(files[1], stream_loc)
        print('device_name = {}, threshold = {}, stream = {}'.format(device_name, threshold, stream))
        csi_path = csi_paths[i]
        data01, pkt1_subs, pkt1_ntx01, pkt1_nrx, nb_pkts = csi_to_data(csi_path)

        Ntx01 = data01['Ntx'][0]
        H01 = data01['H']
        Rssi01 = data01['Rssi'].T

        csi_path = csi_paths[i + 1]
        data02, pkt1_subs, pkt1_ntx02, pkt1_nrx, nb_pkts = csi_to_data(csi_path)
        Ntx02 = data02['Ntx'][0]
        H02 = data02['H']
        Rssi02 = data02['Rssi'].T

        csi_path = csi_paths[i + 2]
        data03, pkt1_subs, pkt1_ntx03, pkt1_nrx, nb_pkts = csi_to_data(csi_path)
        Ntx03 = data03['Ntx'][0]
        H03 = data03['H']
        Rssi03 = data03['Rssi'].T

        str1 = ('|H1| = ({}, {}, {})'.format(len(H01), len(H01[0]), len(H01[0][0])))
        str2 = ('|H2| = ({}, {}, {})'.format(len(H02), len(H02[0]), len(H02[0][0])))
        str3 = ('|H3| = ({}, {}, {})'.format(len(H03), len(H03[0]), len(H03[0][0])))
        print('{}\n{}\n{}'.format(str1, str2, str3))
        min_ = min(len(H01[0][0]), len(H02[0][0]), len(H03[0][0]))
        if len(H01[0][0]) != len(H02[0][0]) != len(H03[0][0]):
            h2 = H02[:, :, 0:min_]
            print(len(h2), len(h2[0]), len(h2[0][0]))
            print('mismatch dimension: only consider first {} streams.'.format(min_))
        data0 = np.concatenate((H01[:, :, 0:min_], H02[:, :, 0:min_], H03[:, :, 0:min_]), axis=1)
        Rssi = np.concatenate((Rssi01, Rssi02, Rssi03), axis=0)
        Ntx = np.hstack([Ntx01, Ntx02, Ntx03])
 #   print('Raw H parts:\n\t{}\n\t{}\n\t{}'.format(str1, str2, str3))
 #   print('|raw_data|: ({}, {}, {}) and |raw Rssi|: ({}, {})'.format(len(data0), len(data0[0]), len(data0[0][0]), len(Rssi), len(Rssi[0])))
    return data0, Rssi, Ntx, device_name, stream, threshold


# This function gets_raw H, Rssi and Ntx data and returns the RssiDropFilter_H, RssiDropFilter_Rssi, RssiDropFilter_Ntx
def get_RssiDropFilter_data(data, Rssi, Ntx):
    RssiShootRow = [int(i) for i in range(len(Rssi)) if any(Rssi[i] <= -820)]
    # print('<= -820: RssiShootRow = {}'.format(RssiShootRow))

    RssiDelta = np.abs(np.diff(Rssi, axis=0))
    # print(RssiDelta)
    RssiDeltaRow = [int(i) for i in range(len(RssiDelta)) if any(RssiDelta[i] >= 50)]
    # print('RssiDelta: RssiDeltaRow = {}'.format(RssiDeltaRow))

    RssiShootIndex = list(set().union(RssiShootRow, RssiDeltaRow))
    # print('RssiShootIndex = \n\t{}'.format(RssiShootIndex))

    RssiDropFilter_H = []
    for sb in range(len(data)):
        d = np.delete(data[sb], RssiShootIndex, axis=0)
        RssiDropFilter_H.append(d)

    RssiDropFilter_Rssi = np.delete(Rssi, RssiShootIndex, axis=0)
    RssiDropFilter_Ntx = np.delete(Ntx, RssiShootIndex, axis=0)
    return RssiDropFilter_H, RssiDropFilter_Rssi, RssiDropFilter_Ntx


# This function reads a data and returns the L2_norm error mean
def get_L2_norm_mean(data, subs, t_window, window_size, stream):
    L2_norms = get_L2_norms(data, subs, t_window, window_size, stream)
    L2_norm_mean = np.mean(L2_norms)
    return L2_norm_mean


def get_L2_norms(data, subs, t_window, window_size, stream):
    t0 = 0
    d0 = [[data[sb][t][stream] for sb in range(subs)] for t in range(t_window)]

    d1 = np.diff(np.transpose(d0), axis=0)

    if 120 <= subs <= 122:
        d1 = np.delete(d1, (subs / 2) - 1, axis=0)

    delta = np.abs(d1)
    L2_norms = np.sqrt(sum(delta ** 2))
    return L2_norms


# This function reads a normalized data and Rssi and returns the best stream
def get_best_stream(data, Rssi, t_window, window_size):
    subs = len(data)
    streams = len(data[0][0])
    # print('\n|data| = ({}, {}, {}), Best stream selection:'.format(subs, len(data[0]), streams))

    # first compute the mean values of the Rssi vector (R0, R1, R2 and R3) within a time window of size 100
    Rssi_mean = np.mean(Rssi[:t_window], axis=0)
    # print('\tRssi_mean: \n\t{}'.format(Rssi_mean))

    if subs == 58:
        # find the indices of the 2 maximum rssi values in Rssi_mean vector:
        max_rssi_indices = Rssi_mean.argsort()[-2:][::-1]
        # print('\tmax_rssi_indices: \n\t{}'.format(max_rssi_indices))

        streams_indices_table = np.reshape(np.arange(16), (4, 4), order='F')

        # mapping to streams indices table and find the top 8 best streams:
        best_streams = streams_indices_table[max_rssi_indices]
        best_streams = np.hstack(best_streams)
        # print('\tbest_8_streams: \n\t{}'.format(best_streams))

        # calculate the L2_norm_mean for each candidate stream in best_streams list and find the best stream according to the minimum L2_norm_mean value:
        L2_norm_means = [get_L2_norm_mean(data, subs, t_window, window_size, stream) for stream in best_streams]
        # print('\tL2_norm_means: \n\t{}'.format(L2_norm_means))

        best_stream = best_streams[L2_norm_means.index(min(L2_norm_means))]
    # print('\tOld data mode: best_stream: {}'.format(best_stream))

    elif subs == 122:
        # mapping to streams indices table and find the 4 streams in Rssi0:
        best_streams = [0, 1, 2, 3]
        # print('\tbest_4_streams: \n\t{}'.format(best_streams))

        # calculate the L2_norm_mean for each candidate stream in best_streams list and find the best stream according to the minimum L2_norm_mean value:
        L2_norm_means = [get_L2_norm_mean(data, subs, t_window, window_size, stream) for stream in best_streams]
        # print('\tL2_norm_means: \n\t{}'.format(L2_norm_means))

        best_stream = best_streams[L2_norm_means.index(min(L2_norm_means))]
    # print('\tNew data mode: best_stream: {}'.format(best_stream))

    return best_stream, best_streams, L2_norm_means


def get_Normalization_sub_st_t(data, subs, stream, time):
    normFactorTarget = 1e6 * 1.6508
    sub_st_t = [data[sb][time][stream] for sb in range(subs)]
    normFactor = np.sqrt((normFactorTarget + 1) / (np.dot(np.transpose(sub_st_t), sub_st_t) + 1))
    for sb in range(subs):
        data[sb][time][stream] *= normFactor
    return data


def get_Normalization_sub_stream_t(data, subs, stream, times):
    for time in np.arange(0, times, 1):
        data = get_Normalization_sub_st_t(data, subs, stream, time)
    return data


def get_normalized_data(data):
    subs = len(data)
    times = len(data[0])
    streams = len(data[0][0])
    # print('input = ({}, {}, {})'.format(subs, times, streams))
    for st in range(streams):
        data = get_Normalization_sub_stream_t(data, subs, st, times)
    return np.array(data)


def get_spiky_packets_index(data, stream, delta_L2_limit=100):
    subs = len(data)
    T = len(data[0])
    d = []
    for t in range(T):
        d.append([data[sb][t][stream] for sb in range(subs)])

    delta = np.abs(np.diff(np.transpose(d), axis=0))
    L2_error = np.sqrt(sum(delta ** 2));

    L2_error_delta = np.abs(np.diff(L2_error))
    L2_error_deltaRow = [i for i in range(len(L2_error_delta)) if L2_error_delta[i] > delta_L2_limit]
    return L2_error_deltaRow


# remove_spiky_packets based on all streams
def get_L2_NormsFiltered_data_v0(data, Rssi, Ntx, delta_L2_limit):
    '''remove spiky packets with L2 norm delta that are above delta_L2_limit
    '''
    streams = len(data[0][0])

    # print('\nApplying L2_NormsFilter:')
    L2_error_deltaRows = []
    for stream in range(streams):
        # print('\tstream = {}'.format(stream))
        L2_error_deltaRow = get_spiky_packets_index(data, stream, delta_L2_limit)
        L2_error_deltaRows.append(L2_error_deltaRow)
    # print('\t\tRemovedIndex = \n\t\t{}'.format(L2_error_deltaRow))

    L2_error_deltaRows2 = np.unique(np.hstack(L2_error_deltaRows))
    # print('L2_error_deltaRows2 = \n\t{}'.format(L2_error_deltaRows2))

    L2_NormsFiltered_data = []
    for sb in range(len(data)):
        d2 = np.delete(data[sb], L2_error_deltaRows2, axis=0)
        L2_NormsFiltered_data.append(d2)

    L2_NormsFiltered_Rssi = np.delete(Rssi, L2_error_deltaRows2, axis=0)
    L2_NormsFiltered_Ntx = np.delete(Ntx, L2_error_deltaRows2, axis=0)
    return np.array(L2_NormsFiltered_data), L2_NormsFiltered_Rssi, L2_NormsFiltered_Ntx


# remove_spiky_packets based on only one stream
def get_L2_NormsFiltered_data_v1(data, Rssi, Ntx, delta_L2_limit, stream):
    '''remove spiky packets with L2 norm delta that are above delta_L2_limit
    '''

    # print('\nApplying L2_NormsFilter:')
    L2_error_deltaRows = []
    L2_error_deltaRow = get_spiky_packets_index(data, stream, delta_L2_limit)
    L2_error_deltaRows.append(L2_error_deltaRow)
    # print('\t\tRemovedIndex = \n\t\t{}'.format(L2_error_deltaRow))

    L2_NormsFiltered_data = []
    for sb in range(len(data)):
        d2 = np.delete(data[sb], L2_error_deltaRows, axis=0)
        L2_NormsFiltered_data.append(d2)

    L2_NormsFiltered_Rssi = np.delete(Rssi, L2_error_deltaRows, axis=0)
    L2_NormsFiltered_Ntx = np.delete(Ntx, L2_error_deltaRows, axis=0)
    return np.array(L2_NormsFiltered_data), L2_NormsFiltered_Rssi, L2_NormsFiltered_Ntx


def get_the_windows(t, shoots, win_L2_Filt):
    for i in shoots:
        if i >= t - win_L2_Filt:
            win_L2_Filt += 1
    return win_L2_Filt


def get_L2_NormsFiltered_data_adaptive(data, Rssi, Ntx, t_start, win_L2_Filt, k, stream0, Zscore_cutoff):
    '''This function applies L2_norms filter on all streams and remove the packets whose L2 z-score are greater than a
    given threshold
    '''

    subs = len(data)
    T = len(data[0])
    streams = len(data[0][0])

    L2_norms = [get_L2_norms(data, subs=subs, t_window=T, window_size=T, stream=stream) for stream in
                range(streams)]

    # find the packet indices that corresponding l2_zscore >= Zscore_cutoff
    # print('\nApplying L2_NormsFilter2 with Zscore_cutoff = {}:'.format(Zscore_cutoff))

    L2_norms_stream0 = L2_norms[stream0]
    # print('\tstream = {}'.format(stream0))

    # print('L2_norms_stream0 = {}'.format(L2_norms_stream0))
    # shoot indices in online form:
    w = win_L2_Filt
    shoots = []
    for t in range(T):
        if t >= win_L2_Filt:

            w = get_the_windows(t, shoots, win_L2_Filt)
            indices = np.arange(t - w, t, 1)

            lst = [i for i in indices if i not in shoots]
            l2 = [L2_norms_stream0[j] for j in lst]  # get L2_norms values of the lst indices
            mu = np.mean(l2)
            sd = np.std(l2, ddof=1)
            l2_zscore = abs(L2_norms_stream0[t] - mu) / sd

            if l2_zscore >= Zscore_cutoff:
                shoots.append(t)

    # print('\t\tRemovedIndex = \n\t\t{}'.format(shoots))
    RemovedIndices = shoots

    # remove found indices from data in batch format:
    L2NormFilter_data = []
    for sb in range(subs):
        d2 = np.delete(data[sb], RemovedIndices, axis=0)
        # print('data[{}] = {}, after: {}'.format(sb, data[sb]))
        L2NormFilter_data.append(d2)

    L2_NormsFiltered_Rssi = np.delete(Rssi, RemovedIndices, axis=0)
    L2_NormsFiltered_Ntx = np.delete(Ntx, RemovedIndices, axis=0)
    return np.array(L2NormFilter_data), L2_NormsFiltered_Rssi, L2_NormsFiltered_Ntx


def get_L2_NormsFiltered_data(data, Rssi, Ntx, t_start, window_size, win_L2_Filt, k, Zscore_cutoff, delta_L2_limit,
                              stream, L2_norm_filter):
    if L2_norm_filter == 'v2':
        return get_L2_NormsFiltered_data_adaptive(data, Rssi, Ntx, t_start, window_size, win_L2_Filt, k, stream,
                                                  Zscore_cutoff)
    elif L2_norm_filter == 'v1':
        return get_L2_NormsFiltered_data_v1(data, Rssi, Ntx, delta_L2_limit, stream)
    elif L2_norm_filter == 'v0':
        return get_L2_NormsFiltered_data_v0(data, Rssi, Ntx, delta_L2_limit)


def get_Filtered2D_data(data, subs, times, streams):
    # This function applies filters with average kernel
    temp = data.reshape((streams * subs * times), order='C')  # pass a Python range and reshape
    temp2 = temp.reshape(subs, times, streams).swapaxes(0, 2)

    kernel2D = np.ones((3, 3)) / 9.
    filtered_data0 = []
    for st in range(len(temp2)):
        filtered2D = signal.convolve2d(np.transpose(temp2[st]), kernel2D, mode='valid')
        filtered_data0.append(np.transpose(filtered2D))

    c1, c2, c3 = len(filtered_data0), len(filtered_data0[0]), len(filtered_data0[0][0])
    temp = (np.array(filtered_data0)).reshape((c1 * c2 * c3), order='C')  # pass a Python range and reshape
    Filtered2D_data = temp.reshape(c1, c2, c3).swapaxes(0, 2)
    return Filtered2D_data


# This function reads a csi filename string and returns the csi specifications
def compile_file_str(file_str, stream_loc):
    file_splitted = file_str.split('_')
    # print('file_splitted: {}'.format(file_splitted))

    device_name0 = file_splitted[0].split('_')[0]
    device_name = device_name0.split('/')[-1]

    stream_code = file_splitted[stream_loc]
    stream = map_stream(stream_code)
    # threshold = thresholds_dict[device_name]
    threshold = thresholds_dict.get(device_name, 15)
    return device_name, stream, threshold


def get_params(samples, gaussian=True):
    if gaussian:
        return get_gaussian_params(samples)
    else:
        return get_statistic_params(samples)


def get_gaussian_params(samples):
    ''' distribution fitting
    now, param[0] and param[1] are the mean and the standard deviation of the fitted distribution
    '''
    param = norm.fit(samples)
    return param[0], param[1]


def get_statistic_params(samples):
    ''' the mean and the standard deviation of the given samples in two passes.
    '''
    mu = np.mean(samples)
    sigma = np.std(samples, ddof=1)
    return mu, sigma


def get_window_samples(d, time, stream, window_size):
    if time < window_size:
        samples = []
        for t in range(time):
            samples.append(d[t][stream])
    else:
        samples = []
        for t in range(time - window_size, time, 1):
            samples.append(d[t][stream])
    return samples


def get_sigmas_subcarriers(data0, time, subcarrier_n, stream, window_size, gaussian):
    sigmas_subcarriers = []
    for subcarrier in range(subcarrier_n):
        d = data0[subcarrier]
        samples_s = get_window_samples(d, time, stream, window_size)
        mu_s, sigma_s = get_params(samples_s, gaussian)
        sigmas_subcarriers.append(sigma_s * sigma_s)
    return sigmas_subcarriers


def get_variance_subcarriers(data0, t, T, subcarrier_n, stream, window_size, gaussian):
    sigmas = []
    while t < T:
        var = get_sigmas_subcarriers(data0, t, subcarrier_n, stream, window_size, gaussian)
        sigmas.append(var)
        t += 1
    return sigmas


class SnaptoCursor(object):
    def __init__(self, ax, x, y):
        self.ax = ax
        self.ly = ax.axvline(color='k', alpha=0.2)  # the vert line
        self.marker, = ax.plot([0], [0], marker="o", color="crimson", zorder=3)
        self.x = x
        self.y = y
        self.txt = ax.text(0.7, 0.9, '')

    def mouse_move(self, event):
        if not event.inaxes: return
        x, y = event.xdata, event.ydata
        indx = np.searchsorted(self.x, [x])[0]
        x = self.x[indx]
        y = self.y[indx]
        self.ly.set_xdata(x)
        self.marker.set_data([x], [y])
        self.txt.set_text('x=%1.2f, y=%1.2f' % (x, y))
        self.txt.set_position((x, y))
        self.ax.figure.canvas.draw_idle()


# This function plots the CSI data for a given stream
def plot_csi(data, stream, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(np.arange(0, np.size(data, axis=0)), np.arange(0, np.size(data, axis=1)))
    Z = np.transpose(np.absolute(data[:, :, stream]))
    # surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    surf = ax.plot_surface(X, Y, Z, cmap='jet_r', linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)  # Add a color bar which maps values to colors.
    plt.title('CSI plot on stream = {} after {}'.format(stream, title))
    plt.show()


# from core.motion import motion_utils
import numpy as np
import math
from random import randint
from matplotlib import cm
from sklearn import preprocessing
from sklearn.cluster import KMeans
import pandas as pd


def check_orangeAlarm(ts, threshold, orangeVotingCount):
    counts = 0
    for i in range(len(ts)):
        if ts[i] > threshold:
            counts += 1
    if counts > orangeVotingCount:
        orangeAlarm = True
    else:
        orangeAlarm = False
    return orangeAlarm


def check_redAlarm(lst):
    return all(lst[i] == lst[i + 1] == True for i in range(len(lst) - 1))


def check_redAlarm2(ts, threshold, redVotingCount):
    counts = 0
    for i in range(len(ts)):
        if ts[i] > threshold:
            counts += 1
    if counts > redVotingCount:
        redAlarm = True
    else:
        redAlarm = False
    return redAlarm


def get_orangeAlarm(ts, threshold, orangeVotingCount):
    return check_orangeAlarm(ts, threshold, orangeVotingCount)


def update_currentQueueDomains(currentQueueDomains, value):
    currentQueueDomains.pop()
    currentQueueDomains.appendleft(value)


def plot_activity_Rssi_L2(data, variances, Rssi, stream, best_streams, window_size80, stride, threshold, k,
                          orangeVotingCount, window_size_r, redVotingCount, L2_norms):
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(j) for j in np.linspace(0, 1, len(best_streams))]

    subcarrier_n = len(data)
    T = len(data[0])
    t = 40
    t_start = t
    subs = len(data)
    streams = len(data[0][0])

    fig, ax = plt.subplots(figsize=(25, 15))
    plt.subplots_adjust(wspace=.35, hspace=.22)

    ax1 = plt.subplot(3, 1, 1)

    subcarrier_n = len(data)
    t = 1
    t_start = 1

    L2_norms_stream = L2_norms[stream]
    activity_level = get_activity_level_original(variances, threshold, k, window_size80, L2_norms_stream)

    linestyles = ['_', '-', '--', ':']
    markers = []
    for m in Line2D.markers:
        try:
            if len(m) == 1 and m != ' ':
                markers.append(m)
        except TypeError:
            pass
    l = len(activity_level)
    plt.plot(range(l), activity_level, color='b')
    plt.plot(range(l), np.ones(l) * threshold, ':', color='green', label='threshold')
    times = np.arange(t_start, l, 1)
    currentOrangeDomains = deque([None, None, None])
    orangeAlarms = []
    redAlarms = []
    oranges = []
    reds = []

    for t in times:
        if t >= window_size80:
            tw = t - window_size80
        else:
            tw = 0
        ts = activity_level[tw:t]
        mx = np.max(ts)
        if t % stride == 0:
            o = get_orangeAlarm(ts, threshold, orangeVotingCount)
            orangeAlarms.append(o)
            currentOrangeDomains.pop()
            currentOrangeDomains.appendleft(o)

            med = np.median(ts)
            if o:
                if med > threshold:
                    ax1.plot(np.arange(tw, t, 1), np.ones(window_size80) * (mx),
                             color='orange', linestyle='--', marker=markers[1], markersize=10)
                    oranges.append((t, activity_level[t]))
            else:
                oranges.append((t, 0))

            Ro1 = all(currentOrangeDomains[i] == currentOrangeDomains[i + 1] == True for i in
                      range(len(currentOrangeDomains) - 1))
            if t >= window_size_r:
                tw_r = t - window_size_r
            else:
                tw_r = 0
            ts_r = activity_level[tw_r:t]
            counts = 0
            for i in range(len(ts_r)):
                if ts_r[i] > threshold:
                    counts += 1
            if counts > redVotingCount:
                Ro2 = True
            else:
                Ro2 = False
            r = Ro1 and Ro2

            redAlarms.append(r)
            if r:
                ax1.plot(np.arange(tw, t, 1), np.ones(window_size80) * (mx), color='red', marker=markers[1])
                reds.append((t, activity_level[t]))
            else:
                reds.append((t, 0))
    motion_windows = [j for i, j in reds if j > 0]
    coverage_rate = (len(motion_windows) / len(reds)) * 100.

    ax1.set_xlabel('Time', fontsize=15)
    ax1.set_ylabel('Activity Level', fontsize=25)
    ax1.legend(loc='upper left')
    str1 = ('# of windows = {}, # of motion windows = {}\n'.format(len(reds), len(motion_windows)))
    str2 = ('coverage_rate = %2.2f %%' % (coverage_rate))
    title = str1 + str2
    ax1.set_title('Activity level (Original) and motion detection on stream = {}\n{}'.format(stream, title),
                  fontsize=25)

    colors2 = ['b', 'orange', 'g', 'r']
    Rssi = pd.DataFrame(Rssi)
    ax2 = plt.subplot(3, 1, 2)
    for i in range(4):
        c = colors2[i]
        col = 'R{}'.format(i)
        ax2.plot(Rssi.iloc[:, i], c=c, label='Rssi {}'.format(i))
        i += 1
    ax2.set_xlabel('Time', fontsize=15)
    ax2.set_ylabel('Rssi', fontsize=25)
    ax2.legend(loc='upper left')

    ax3 = plt.subplot(3, 1, 3)
    i = 0
    for s in best_streams:
        c = colors[i]
        ax3.plot(L2_norms[s], marker=markers[i], markersize=5, c=c, label='stream {}'.format(s))
        i += 1
    ax3.set_xlabel('Time', fontsize=15)
    ax3.set_ylabel('L2_norm', fontsize=25)
    ax3.legend(loc='upper left')

    plt.show()
    return oranges, reds


# This function calculates the original activity level without considering baseline
def get_activity_level_original(variances, threshold, k, window_size, L2_norms_stream):
    n = len(variances)
    magMinVarsMeanRaws = []
    activity_level = []
    ss, sds, mus, cis, meds, mns, mxs, sd_diffs, l2_sd_diffs = [], [], [], [], [], [], [], [], []
    w2 = int(window_size / 2)
    for i in range(n):
        magMinVarsMeanRaw = 2.5 * (np.mean(sorted(variances[i], reverse=False)[:k]))
        magMinVarsMeanRaws.append(magMinVarsMeanRaw)

        last_window = magMinVarsMeanRaws[-window_size:]
        mu = np.mean(last_window)
        c_threshold = ([1 for c in last_window if c >= threshold]).count(1)
        c_mu = sum(1 for c in last_window if c > mu)
        activity_level.append(magMinVarsMeanRaw)
    return activity_level


# This function calculates the standard activity level
def get_activity_level_standard(variances, threshold, k, window_size, L2_norms_stream):
    n = len(variances)
    magMinVarsMeanRaws = []
    activity_level = []
    w2 = int(window_size / 2)
    dir_path = '/home/farnoush/Aerial.ai/Projects/csi_data/test_negar/'

    for i in range(n):
        magMinVarsMeanRaw = 2.5 * (np.mean(sorted(variances[i], reverse=False)[:k]))
        magMinVarsMeanRaws.append(magMinVarsMeanRaw)
        if i > window_size:
            # adapting baseline:
            last_window = magMinVarsMeanRaws[-window_size:]
            l2 = L2_norms_stream[i - w2:i]
            l2_sd_diff = np.std(np.diff(l2, axis=0))

            mn = np.min(last_window)
            mx = np.max(last_window)
            mu = np.mean(last_window)
            med = np.median(last_window)
            sd = np.std(last_window)
            s = sd ** (0.5)
            c_threshold = ([1 for c in last_window if c >= threshold]).count(1)
            c_mu = sum(1 for c in last_window if c > mu)

            baseline = min(25, mn)
            activity_level.append(magMinVarsMeanRaws[i] - baseline)
        else:
            activity_level.append(magMinVarsMeanRaw)
    return activity_level


def plot_activity(activity_level, activity, window_size, t_window, stride, threshold,
                  orangeVotingCount, window_size_r, redVotingCount, stream):
    cmap = plt.get_cmap('gnuplot')

    fig, ax = plt.subplots(figsize=(25, 15))
    plt.subplots_adjust(wspace=.35, hspace=.22)

    ax1 = plt.subplot(3, 1, 1)
    t = 1
    t_start = 1
    T = len(activity_level)
    linestyles = ['_', '-', '--', ':']
    markers = []
    for m in Line2D.markers:

        try:
            if len(m) == 1 and m != ' ':
                markers.append(m)
        except TypeError:
            pass
    activity_colors = {'original': 'b', 'standard': 'green', 'adaptive_ARIMA': 'aqua'}
    plt.plot(range(len(activity_level)), activity_level, color=activity_colors[activity])  # 'aqua')
    plt.plot(np.arange(t_start, T, 1), np.ones(T - t_start) * threshold, ':', color='black', label='threshold')
    times = np.arange(t_start, T, 1)

    currentOrangeDomains = deque([None, None, None])
    orangeAlarms = []
    redAlarms = []
    oranges = []
    reds = []
    for t in times:
        if t >= window_size:
            tw = t - window_size
        else:
            tw = 0
        ts = activity_level[tw:t]
        mean = np.mean(ts)
        mx = np.max(ts)

        if t % stride == 0:
            o = get_orangeAlarm(ts, threshold, orangeVotingCount)
            orangeAlarms.append(o)
            currentOrangeDomains.pop()
            currentOrangeDomains.appendleft(o)

            med = np.median(ts)

            if o:
                if med > threshold:
                    ax1.plot(np.arange(tw, t, 1), np.ones(window_size) * (mx),
                             color='orange', linestyle='--', marker=markers[1], markersize=10)
                    oranges.append((t, activity_level[t]))
            else:
                oranges.append((t, 0))

            Ro1 = all(currentOrangeDomains[i] == currentOrangeDomains[i + 1] == True for i in
                      range(len(currentOrangeDomains) - 1))
            if t >= window_size_r:
                tw_r = t - window_size_r
            else:
                tw_r = 0
            ts_r = activity_level[tw_r:t]
            counts = 0
            for i in range(len(ts_r)):
                if ts_r[i] > threshold:
                    counts += 1
            if counts > redVotingCount:
                Ro2 = True
            else:
                Ro2 = False

            r = Ro1 and Ro2

            redAlarms.append(r)
            if r:
                ax1.plot(range(tw, t, 1), np.ones(window_size) * (mx), color='red', marker=markers[1])
                reds.append((t, r))
            else:
                reds.append((t, 0))
    motion_windows = [j for i, j in reds if j == True]
    coverage_rate = float(len(motion_windows) / len(reds)) * 100.

    ax1.set_xlabel('Time', fontsize=15)
    ax1.set_ylabel('Activity Level', fontsize=25)
    ax1.legend(loc='upper left')
    str1 = ('# of windows = {}, # of motion windows = {}\n'.format(len(reds), len(motion_windows)))
    str2 = ('coverage_rate = %2.2f %%' % (coverage_rate))
    title = str1 + str2
    ax1.set_title('Activity level_{} and motion detection on stream = {}\n{}'.format(activity, stream, title),
                  fontsize=25)
    plt.show()


def main(files, loc, t_window, t, k, t_start, window_size80,
         win_L2_Filt, Zscore_cutoff, delta_L2_limit, L2_norm_filter, kernel2D,
         threshold2):
    # Reading everything from files:
    csi_pathes = filepaths_from_files(files)
    data0, Rssi, Ntx, device_name, stream, threshold = get_data_Rssi_Ntx_from_files(csi_pathes, files, stream_loc=loc)

    # Applying RssiDropFilter:
    RssiDropFilter_H, RssiDropFilter_Rssi, RssiDropFilter_Ntx = get_RssiDropFilter_data(data0, Rssi, Ntx)
    RssiDropFilter_data = np.absolute(RssiDropFilter_H)
    # print('|RssiDropFilter_data|: ({}, {}, {})'.format(len(RssiDropFilter_data), len(RssiDropFilter_data[0]),
    #                                                    len(RssiDropFilter_data[0][0])))
    # print('|RssiDropFilter_Rssi|: ({}, {})'.format(len(RssiDropFilter_Rssi), len(RssiDropFilter_Rssi[0])))
    # print('|RssiDropFilter_Ntx|: ({})'.format(len(RssiDropFilter_Ntx)))

    plot_csi(RssiDropFilter_data, stream=stream, title='RssiDropFilter_data')

    # Data Normalization:
    Normalized_data = get_normalized_data(RssiDropFilter_data)
    # print('|Normalized_data|: ({}, {}, {})'.format(len(Normalized_data), len(Normalized_data[0]),
    #                                                len(Normalized_data[0][0])))
    plot_csi(Normalized_data, stream=stream, title='Normalized_data')

    # Applying L2_NormsFilter:
    L2_NormsFiltered_data, L2_NormsFiltered_Rssi, L2_NormsFiltered_Ntx = get_L2_NormsFiltered_data_adaptive(
        Normalized_data, RssiDropFilter_Rssi, RssiDropFilter_Ntx,
        t_start, win_L2_Filt, k, stream0=0, Zscore_cutoff=Zscore_cutoff)
    # print('L2_NormsFiltered_data: ({}, {}, {})'.format(len(L2_NormsFiltered_data), len(L2_NormsFiltered_data[0]),
    #                                                    len(L2_NormsFiltered_data[0][0])))

    # Best Stream Selection:
    best_stream, best_8_streams, L2_norm_means = get_best_stream(L2_NormsFiltered_data, L2_NormsFiltered_Rssi,
                                                                 t_window, window_size80)
    L2_norms = [get_L2_norms(L2_NormsFiltered_data, subs=len(L2_NormsFiltered_data),
                             t_window=len(L2_NormsFiltered_data[0]), window_size=len(L2_NormsFiltered_data[0]),
                             stream=stream) for stream in range(len(L2_NormsFiltered_data[0][0]))]
    L2_norms_stream = L2_norms[best_stream]

    # Applying Filtered2D:
    Filtered2D_data = get_Filtered2D_data(L2_NormsFiltered_data, subs=len(L2_NormsFiltered_data),
                                          times=len(L2_NormsFiltered_data[0]),
                                          streams=len(L2_NormsFiltered_data[0][0]))

    plot_csi(Filtered2D_data, stream=best_stream, title='Filtered2D_data')
    # print('|Filtered2D_data|: ({}, {}, {})'.format(len(Filtered2D_data), len(Filtered2D_data[0]),
    #                                                len(Filtered2D_data[0][0])))

    # Compute Variance:
    subs = len(Filtered2D_data)
    T = len(Filtered2D_data[0])
    streams = len(Filtered2D_data[0][0])

    t = 1
    t_start = 1
    variances = get_variance_subcarriers(Filtered2D_data, t, T, subcarrier_n=subs, stream=best_stream,
                                         window_size=window_size80, gaussian=False)

    # Activity level:
    activity_level_original = get_activity_level_original(variances, threshold, k, window_size80, L2_norms_stream)
    activity_level_standard = get_activity_level_standard(variances, threshold, k, window_size80, L2_norms_stream)

    # plot all activities in a single graph:
    fig, ax = plt.subplots(figsize=(20, 4))
    fig.subplots_adjust(wspace=.4, hspace=.75)

    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(activity_level_original, color='b', label='Original')
    ax1.plot(range(len(activity_level_standard[1:])), activity_level_standard[1:], color='green', label='Standard')
    ax1.plot(range(len(activity_level_original)), np.ones(len(activity_level_original)) * threshold, ':',
             color='black', label='Threshold')
    ax1.legend()
    plt.show()


if __name__ == "__main__":
    captures = [['Motions/84AA9C0BB0D5_Motion_Armed_0040_2018-06-12_17h20_32_P1.csi',
                 'Motions/84AA9C0BB0D5_Motion_Armed_0040_2018-06-12_17h20_32_P2.csi',
                 'Motions/84AA9C0BB0D5_Motion_Armed_0040_2018-06-12_17h20_32_P3.csi'],
                ]

    t_window = 100
    t = 1
    k = 40
    t_start = 1
    win_L2_Filt = 40
    window_size80 = 80
    Zscore_cutoff = 3
    delta_L2_limit = 100
    L2_norm_filter = 'v2'  # 'v1'
    kernel2D = np.ones((3, 3)) / 9.

    threshold2 = 25

    files = captures[0]
    loc = 3  # 3  4

    main(files, loc, t_window, t, k, t_start, window_size80,
         win_L2_Filt, Zscore_cutoff, delta_L2_limit, L2_norm_filter, kernel2D, threshold2=threshold2)