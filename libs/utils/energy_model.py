# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2015, ARM Limited and contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import logging
import matplotlib.pyplot as plt
import os
import pandas as pd

from collections import namedtuple
from csv import DictWriter
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from time import sleep
from trappy.plotter.ColorMap import ColorMap

# The EM reports capacity and energy consumption for each frequency domain.
# The frequency domains to be considered by the following EM building flow
# are described by the parameters of this named tuple
ClusterDescription = namedtuple('ClusterDescription',
                                [   # Name of the cluster
                                    'name',
                                    # Name of the energy meter channel as
                                    # specified in the target configuration
                                    'emeter_ch',
                                    # Name of the cores in the cluster
                                    'core_name',
                                    # List of cores in the cluster (core IDs)
                                    'cpus',
                                    # List of frequencies to profile
                                    'freqs',
                                    # List of idle states to profile
                                    'idle_states'
                                ])

WFI = 0
CORE_OFF = 1

class EnergyModel(object):
    """
    Energy Model Building Flow API

    :param te: instance of devlib Android target
    :type te: env.TestEnv

    :param clusters_description: a list of `namedtuple(ClusterDescription)`
        objects that describe the topology of the system and provides
        information to the energy model building flow
    :type clusters_description: list(namedtuple(ClusterDescription))

    :param res_dir: path to results directory
    :type res_dir: str
    """

    def __init__(self, te, res_dir):
        self._te = te
        self.clusters = clusters_description
        self._res_dir = res_dir

    def cluster_energy_compute_power_perf(loop_cnt, benchmark,
                                          bkp_file='pstates.csv'):
        """
        Perform P-States profiling on each input cluster for platforms that
        allow energy measurement at cluster level.

        This method requires a `benchmark` object with the following
        characteristics:

            - duration, attribute that tells the workload duration in seconds
            - run(cgroup, threads), run the benchmark into the specified
                                    'cgroup', spawning the specified number of
                                    'threads', and return a performance score
                                    of their execution.

        Data will be saved into a CSV file at each iteration such that, if
        something goes wrong, the user can restart the experiment considering
        only idle_states that had not yet been profiled.

        :param loop_cnt: number of iterations for each experiment
        :type loop_cnt: int

        :param benchmark: benchmark object
# TODO: define a type for benchmark
        :type benchmark: int

        :param bkp_file: CSV file name
        :type bkp_file: str
        """

        target = self._te.target

        # Make sure all CPUs are online
        target.hotplug.online_all()

        # Set cpufreq governor to userpace to allow manual frequency scaling
        target.cpufreq.set_all_governors('userspace')

        bkp_file = os.path.join(self._res_dir, bkp_file)
        with open(bkp_file, 'w') as csvfile:
            writer = DictWriter(csvfile,
                                fieldnames=['cluster', 'cpus', 'freq',
                                            'perf', 'energy', 'power'])

            # A) For each cluster (i.e. frequency domain) to profile...
            power_perf = []
            for cl in self.clusters:
                target_cg, _ = target.cgroups.isolate(cl.cpus)

                # P-States profiling requires to plug in CPUs one at the time
                for cpu in cl.cpus:
                    target.hotplug.offline(cpu)

                # B) For each additional cluster's plugged in CPU...
                on_cpus = []
                for cnt, cpu in enumerate(cl.cpus):

                    # Hotplug ON one more CPU
                    target.hotplug.online(cpu)
                    on_cpus.append(cpu)

                    # Ensure online CPUs are part of the target cgroup
                    # (in case hotplug OFF removes it)
                    target_cg.set(cpus=on_cpus)
                    cl_cpus = set(target.list_online_cpus()).intersection(set(cl.cpus))
                    logging.info('Cluster {:8} (Online CPUs : {})'\
                                  .format(cl.name, list(cl_cpus)))

                    # C) For each OPP supported by the current cluster
                    for freq in cl.freqs:

                        # Set frequency to freq for current CPUs
                        target.cpufreq.set_frequency(cpu, freq)

                        # Run the benchmark for the specified number of
                        # iterations each time collecting a sample of energy
                        # consumption and reported performance
                        energy = 0
                        perf = 0
                        for i in xrange(loop_cnt):
                            self._te.emeter.reset()
                            # Run benchmark into the target cgroup
                            perf += benchmark.run(target_cg.name, cnt + 1)
                            nrg = self._te.emeter.report(self._res_dir).channels
                            energy += nrg[cl.emeter_ch]
                            sleep(10)

                        # Compute average energy and performance for the
                        # current number of active CPUs all running at the
                        # current OPP
                        perf = perf / loop_cnt
                        energy = energy / loop_cnt
                        power = energy / benchmark.duration
                        logging.info('  avg_prf: {:7.3}, avg_pwr: {:7.3}'
                                     .format(perf, power))

                        # Keep track of this new P-State profiling point
                        new_row = {'cluster': cl.name,
                                   'cpus': cnt + 1,
                                   'freq': freq,
                                   'perf': perf,
                                   'energy' : energy,
                                   'power': power}
                        power_perf.append(new_row)

                        # Save data in a CSV file
                        writer.writerow(new_row)

                    # C) profile next P-State

                # B) add one more CPU (for the current frequency domain)

            # A) Profile next cluster (i.e. frequency domain)

            target.hotplug.online_all()

            power_perf_df = pd.DataFrame(power_perf)

        return power_perf_df.set_index(['cluster', 'freq', 'cpus'])\
                            .sort_index(level='cluster')

    def power_perf_stats(power_perf_df):
        """
        For each cluster compute per-OPP power and performance statistics.

        :param power_perf_df: dataframe containing power and performance
            numbers
        :type power_perf_df: :mod:`pandas.DataFrame`
        """
        clusters = power_perf_df.index.get_level_values('cluster')\
                                      .unique().tolist()

        stats = []
        for cl in clusters:
            cl_power_df = power_perf_df.loc[cl].reset_index()

            grouped = cl_power_df.groupby('freq')
            for freq, df in grouped:
                perf = df['perf'] / df['cpus']
                power = df['power'] / df['cpus']
                energy = df['energy'] / df['cpus']

                avg_row = {'cluster': cl,
                           'freq': freq,
                           'stats': 'avg',
                           'perf': perf.mean(),
                           'power': power.mean(),
                           'energy': energy.mean()
                          }
                std_row = {'cluster': cl,
                           'freq': freq,
                           'stats': 'std',
                           'perf': perf.std(),
                           'power': power.std(),
                           'energy': energy.std()
                          }
                min_row = {'cluster': cl,
                           'freq': freq,
                           'stats': 'min',
                           'perf': perf.min(),
                           'power': power.min(),
                           'energy': energy.min()
                          }
                max_row = {'cluster' : cl,
                           'freq' : freq,
                           'stats' : 'max',
                           'perf' : perf.max(),
                           'power' : power.max(),
                           'energy': energy.max()
                          }
                c99_row = {'cluster' : cl,
                           'freq' : freq,
                           'stats' : 'c99',
                           'perf' : perf.quantile(q=0.99),
                           'power' : power.quantile(q=0.99),
                           'energy': energy.quantile(q=0.99)
                          }

                stats.append(avg_row)
                stats.append(std_row)
                stats.append(min_row)
                stats.append(max_row)
                stats.append(c99_row)

        stats_df = pd.DataFrame(stats).set_index(['cluster', 'freq', 'stats'])\
                                      .sort_index(level='cluster')
        return stats_df.unstack()

    def compute_idle_power(loop_cnt, sleep_duration,
                           bkp_file='cstates.csv'):
        """
        Perform C-States profiling on each input cluster.

        Data will be saved into a CSV file at each iteration such that if
        something goes wrong the user can restart the experiment considering
        only idle_states that had not been processed.

        :param loop_cnt: number of loops for each experiment
        :type loop_cnt: int

        :param sleep_duration: sleep time in seconds
        :type sleep_duration: int

        :param bkp_file: CSV file name
        :type bkp_file: str
        """

        target = self._te.target

        # Make sure all CPUs are online
        target.hotplug.online_all()

        with open(bkp_file, 'w') as csvfile:
            writer = DictWriter(csvfile, fieldnames=['cluster', 'cpus',
                                            'idle_state', 'energy', 'power'])

            # Disable frequency scaling by setting cpufreq governor to userspace
            target.cpufreq.set_all_governors('userspace')

            # A) For each cluster (i.e. frequency domain) to profile...
            idle_power = []
            for cl in self.clusters:
                target.cgroups.isolate(cl.cpus)

                # C-States profiling requires to plug in CPUs one at the time
                for cpu in cl.cpus:
                    target.hotplug.offline(cpu)

                # B) For each additional cluster's plugged in CPU...
                for cnt, cpu in enumerate(cl.cpus):

                    # Hotplug ON one more CPU
                    target.hotplug.online(cpu)

                    cl_cpus = set(target.list_online_cpus()).intersection(set(cl.cpus))
                    logging.info('Cluster {:8} (Online CPUs : {})'\
                                 .format(cl.name, list(cl_cpus)))

                    # C) For each OPP supported by the current cluster
                    for idle in cl.idle_states:

                        # Disable all idle states but the current one
                        for c in cl.cpus:
                            target.cpuidle.disable_all(cpu=c)
                            target.cpuidle.enable(idle, cpu=c)

                        # Sleep for the specified duration each time
                        # collecting a sample of energy consumption and
                        # reported performance
                        energy = 0
                        for i in xrange(loop_cnt):
                            self._te.emeter.reset()
                            sleep(sleep_duration)
                            nrg = self._te.emeter.report(te.res_dir).channels
                            energy += nrg[cl.emeter_ch]

                        # Compute average energy and performance for the
                        # current number of active CPUs all idle at the
                        # current OPP
                        energy = energy / loop_cnt
                        power = energy / SLEEP_DURATION
                        logging.info('  avg_pwr: {:7.3}'.format(power))

                        # Keep track of this new C-State profiling point
                        new_row = {'cluster': cl.name,
                                   'cpus': cnt + 1,
                                   'idle_state': idle,
                                   'energy': energy,
                                   'power': power}
                        idle_power.append(new_row)

                        # Save data in a CSV file
                        writer.writerow(new_row)

                    # C) profile next C-State

                # B) add one more CPU (for the current frequency domain)

            # A) profile next cluster (i.e. frequency domain)

            target.hotplug.online_all()

            idle_df = pd.DataFrame(idle_power)
        return idle_df.set_index(
            ['cluster', 'idle_state', 'cpus']).sort_index(level='cluster')

    def idle_power_stats(idle_df):
        """
        For each cluster compute per idle state power statistics.

        :param idle_df: dataframe containing power numbers
        :type idle_df: :mod:`pandas.DataFrame`
        """

        stats = []
        for cl in self.clusters:
            cl_df = idle_df.loc[cl.name].reset_index()
            # Start from deepest idle state
            cl_df = cl_df.sort_values('idle_state', ascending=False)
            grouped = cl_df.groupby('idle_state', sort=False)
            for state, df in grouped:
                energy = df.energy
                power = df.power
                state_name = "C{}_CLUSTER".format(state)
                if state == CORE_OFF:
                    core_off_nrg_avg = energy.mean()
                    core_off_pwr_avg = power.mean()
                if state == WFI:
                    energy = df.energy.diff()
                    energy[0] = df.energy[0] - core_off_nrg_avg
                    power = df.power.diff()
                    power[0] = df.power[0] - core_off_pwr_avg
                    state_name = "C0_CORE"

                avg_row = {'cluster': cl.name,
                           'idle_state': state_name,
                           'stats': 'avg',
                           'energy': energy.mean(),
                           'power': power.mean()
                          }
                std_row = {'cluster': cl.name,
                           'idle_state': state_name,
                           'stats': 'std',
                           'energy': energy.std(),
                           'power': power.std()
                          }
                min_row = {'cluster' : cl.name,
                           'idle_state' : state_name,
                           'stats' : 'min',
                           'energy' : energy.min(),
                           'power' : power.min()
                          }
                max_row = {'cluster' : cl.name,
                           'idle_state' : state_name,
                           'stats' : 'max',
                           'energy' : energy.max(),
                           'power' : power.max()
                          }
                c99_row = {'cluster' : cl.name,
                           'idle_state' : state_name,
                           'stats' : 'c99',
                           'energy' : energy.quantile(q=0.99),
                           'power' : power.quantile(q=0.99)
                          }
                stats.append(avg_row)
                stats.append(std_row)
                stats.append(min_row)
                stats.append(max_row)
                stats.append(c99_row)

        stats_df = pd.DataFrame(stats).set_index(
            ['cluster', 'idle_state', 'stats']).sort_index(level='cluster')
        return stats_df.unstack()

    def pstates_model_df(pp_stats, power_perf_df, metric='avg'):
        """
        Build two data frames containing data to create the energy model for each
        cluster given as input.

        :param pp_stats: power and performance statistics
        :type pp_stats: :mod:`pandas.DataFrame`

        :param power_perf_df: power and performance data
        :type power_perf_df: :mod:`pandas.DataFrame`
        """
        max_score = pp_stats.perf[metric].max()

        core_cap_energy = []
        cluster_cap_energy = []
        for cl in self.clusters:
            # ACTIVE Energy
            grouped = power_perf_df.loc[cl.name].groupby(level='freq')
            for freq, df in grouped:
                # Get average energy at OPP freq for 1 CPU
                energy_freq_1 = pp_stats.loc[cl.name].loc[freq]['energy'][metric]
                # Get cluster energy at OPP freq
                x = df.index.get_level_values('cpus').tolist()
                y = df.energy.tolist()
                slope, intercept = linfit(x, y)
                # Energy can't be negative but the regression line may intercept the
                # y-axis at a negative value. Im this case cluster energy can be
                # assumed to be 0.
                cluster_energy = intercept if intercept >= 0.0 else 0.0
                core_energy = energy_freq_1 - cluster_energy

                # Get score at OPP freq
                score_freq = pp_stats.loc[cl.name].loc[freq]['perf'][metric]
                capacity = int(score_freq * 1024 / max_score)

                core_cap_energy.append({'cluster' : cl.name,
                                        'core': cl.core_name,
                                        'freq': freq,
                                        'cap': capacity,
                                        'energy': core_energy})
                cluster_cap_energy.append({'cluster': cl.name,
                                           'freq': freq,
                                           'cap': capacity,
                                           'energy': cluster_energy})

        core_cap_nrg_df = pd.DataFrame(core_cap_energy)
        cluster_cap_nrg_df = pd.DataFrame(cluster_cap_energy)
        return core_cap_nrg_df, cluster_cap_nrg_df

    def energy_model_dict(core_cap_nrg_df, cluster_cap_nrg_df, metric='avg'):
        """
        """
        n_states = len(self.clusters[0].idle_states)

        nrg_dict = {}

        grouped = core_cap_nrg_df.groupby('cluster')
        for cl, df in grouped:
            nrg_dict[cl] = {
                "opps" : {},
                "core": {
                    "name": df.core.iloc[0],
                    "busy-cost": OrderedDict(),
                    "idle-cost": OrderedDict()
                },
                "cluster": {
                    "busy-cost": OrderedDict(),
                    "idle-cost": OrderedDict()
                }
            }
            # Core COSTS
            # ACTIVE costs
            for row in df.iterrows():
                nrg_dict[cl]["opps"][row[1].cap] = row[1].freq
                nrg_dict[cl]["core"]["busy-cost"][row[1].cap] = int(row[1].energy)

            # IDLE costs
            wfi_nrg = idle_stats.loc[cl].energy[metric][0]
            # WFI
            nrg_dict[cl]["core"]["idle-cost"][0] = int(wfi_nrg)
            # All remaining states are zeroes
            for i in xrange(1, n_states):
                nrg_dict[cl]["core"]["idle-cost"][i] = 0

            # Cluster COSTS
            cl_data = cluster_cap_nrg_df[cluster_cap_nrg_df.cluster == cl]
            # ACTIVE costs
            for row in cl_data.iterrows():
                nrg_dict[cl]["cluster"]["busy-cost"][row[1].cap] = int(row[1].energy)

            # IDLE costs
            # Core OFF is the first valid idle cost for cluster
            idle_data = idle_stats.loc[cl].energy[metric]
            # WFI (same as Core OFF)
            nrg_dict[cl]["cluster"]["idle-cost"][0] = int(idle_data[1])
            # All other idle states (from CORE OFF down)
            for i in xrange(1, n_states):
                nrg_dict[cl]["cluster"]["idle-cost"][i] = int(idle_data[i])

        return nrg_dict

################################################################################
# Plotting methods
################################################################################

    def plot_pstates(power_perf_df, cluster):
        """
        Plot P-States profiling for the specified cluster.

        :param power_perf_df: DataFrame reporting power and performance values
        :type power_perf_df: :mod:`pandas.DataFrame`

        :param cluster: cluster description
        :type cluster: namedtuple(ClusterDescription)
        """
        cmap = ColorMap(len(cluster.freqs))
        color_map = map(cmap.cmap, range(len(cluster.freqs)))
        color_map = dict(zip(cluster.freqs, color_map))

        fig, ax = plt.subplots(1, 1, figsize=(16, 10))

        grouped = power_perf_df.loc[cluster.name].groupby(level='freq')
        for freq, df in grouped:
            x = df.index.get_level_values('cpus').tolist()
            y = df.power.tolist()
            slope, intercept = linfit(x, y)
            x.insert(0, 0)
            y.insert(0, intercept)
            # Plot linear fit of the points
            ax.plot(x, [slope*i + intercept for i in x], color=color_map[freq])
            # Plot measured points
            ax.scatter(x, y, color=color_map[freq], label='{} kHz'.format(freq))

        ax.set_title('{} cluster P-States profiling'.format(cluster.name),
                     fontsize=16)
        ax.legend()
        ax.set_xlabel('Active cores')
        ax.set_ylabel('Power [$\mu$W]')
        ax.set_xlim(-0.5, len(cluster.cpus)+1)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.grid(True)

    def plot_power_perf(pp_stats):
        """
        Plot Power VS performance curves for the input clusters

        :param power_perf_df: DataFrame reporting power and performance values
        :type power_perf_df: :mod:`pandas.DataFrame`
        """
        cmap = ColorMap(len(self.clusters) + 1)
        color_map = map(cmap.cmap, range(len(self.clusters) + 1))

        fig, ax = plt.subplots(1, 1, figsize=(16, 10))

        max_perf = pp_stats.perf['avg'].max()
        max_power = pp_stats.power['avg'].max()

        for i, cl in enumerate(self.clusters):
            cl_df = pp_stats.loc[cl.name]
            norm_perf_df = cl_df.perf['avg'] * 100.0 / max_perf
            norm_power_df = cl_df.power['avg'] * 100.0 / max_power

            x = norm_perf_df.values.tolist()
            y = norm_power_df.values.tolist()
            ax.plot(x, y, color=color_map[i], marker='o', label=cl.name)

            norm_perf_df = cl_df.perf['max'] * 100.0 / max_perf
            norm_power_df = cl_df.power['max'] * 100.0 / max_power

            x = norm_perf_df.values.tolist()
            y = norm_power_df.values.tolist()
            ax.plot(x, y, '--', color=color_map[-1])

            norm_perf_df = cl_df.perf['min'] * 100.0 / max_perf
            norm_power_df = cl_df.power['min'] * 100.0 / max_power

            x = norm_perf_df.values.tolist()
            y = norm_power_df.values.tolist()
            ax.plot(x, y, '--', color=color_map[-1])

        ax.set_title('Power VS Performance curves', fontsize=16)
        ax.legend()
        ax.set_xlabel('Performance [%]')
        ax.set_ylabel('Power [%]')
        ax.set_xlim(0, 120)
        ax.set_ylim(0, 120)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True)

    def plot_cstates(idle_power_df, cluster):
        """
        Plot C-States profiling for the specified cluster.

        :param idle_power_df: dataframe reporting power values in each idle
            state
        :type idle_power_df: :mod:`pandas.DataFrame`

        :param cluster: cluster description
        :type cluster: namedtuple(ClusterDescription)
        """
        n_cpus = len(cluster.cpus)
        cmap = ColorMap(len(cluster.idle_states))
        color_map = map(cmap.cmap, cluster.idle_states)
        color_map = [c for c in color_map for i in xrange(n_cpus)]

        cl_df = idle_power_df.loc[cluster.name]
        ax = cl_df.power.plot.bar(figsize=(16,8), color=color_map, alpha=0.5,
                                  legend=False, table=True)

        idx = 0
        grouped = cl_df.groupby(level=0)
        for state, df in grouped:
            x = df.index.get_level_values('cpus').tolist()
            y = df.power.tolist()
            slope, intercept = linfit(x, y)

            y = [slope * v + intercept for v in x]
            x = range(n_cpus * idx, n_cpus * (idx + 1))
            ax.plot(x, y, color=color_map[idx*n_cpus], linewidth=4)
            idx += 1

        ax.grid(True)
        ax.get_xaxis().set_visible(False)
        ax.set_ylabel("Idle Power [$\mu$W]")
        ax.set_title("{} cluster C-states profiling"\
                     .format(cluster.name), fontsize=16)


################################################################################
# Methods to generate Energy Model output files
################################################################################



# vim :set tabstop=4 shiftwidth=4 expandtab
