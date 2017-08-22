#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import random
import os
from scipy.misc import imsave
import gym

from build_model import LatentRL
# from data_manager import DataManager

tf.app.flags.DEFINE_float("beta", 4.0, "beta parameter for latent loss")
tf.app.flags.DEFINE_integer("epoch_size", 2000, "epoch size")
tf.app.flags.DEFINE_integer("batch_size", 64, "batch size")
tf.app.flags.DEFINE_float("learning_rate", 1e-2, "learning rate")
tf.app.flags.DEFINE_string("checkpoint_dir", "checkpoints", "checkpoint directory")
tf.app.flags.DEFINE_string("log_file", "./log", "log file directory")
tf.app.flags.DEFINE_boolean("training", True, "training or not")

flags = tf.app.flags.FLAGS

SUMMARY_INTERVAL = 100


  




def train(sess,
          model,
          manager,
          saver,
          display_step=1):

  tf.summary.scalar("loss", model.loss)
  summary_op = tf.summary.merge_all()
  summary_writer = tf.summary.FileWriter(flags.log_file,
                                         sess.graph)

  n_samples = manager.sample_size

  reconstruct_check_images = manager.get_random_images(10)

  indices = range(n_samples)

  step = 0
  
  # Training cycle
  for epoch in range(flags.epoch_size):
    # Shuffle image indices
    random.shuffle(indices)
    
    avg_cost = 0.0
    total_batch = n_samples // flags.batch_size
    
    # Loop over all batches
    for i in range(total_batch):
      # Generate image batch
      batch_indices = indices[flags.batch_size*i : flags.batch_size*(i+1)]
      batch_xs = manager.get_images(batch_indices)
      
      # Fit training using batch data
      if step % SUMMARY_INTERVAL == SUMMARY_INTERVAL-1:
        cost, summary_str = model.partial_fit(sess, batch_xs, summary_op)
        summary_writer.add_summary(summary_str, step)
      else:
        cost = model.partial_fit(sess, batch_xs)

      # Compute average loss
      avg_cost += cost / n_samples * flags.batch_size
      step += 1

     # Display logs per epoch step
    if epoch % display_step == 0:
      print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    # Image reconstruction check
    reconstruct_check(sess, model, reconstruct_check_images)

    # Disentangle check
    disentangle_check(sess, model, manager)

    # Save checkpoint
    saver.save(sess, flags.checkpoint_dir + '/' + 'checkpoint', global_step = epoch)

    
def reconstruct_check(sess, model, images):
  # Check image reconstruction
  x_reconstruct = model.reconstruct(sess, images)

  if not os.path.exists("reconstr_img"):
    os.mkdir("reconstr_img")

  for i in range(len(images)):
    org_img = images[i].reshape(64, 64)
    org_img = org_img.astype(np.float32)
    reconstr_img = x_reconstruct[i].reshape(64, 64)
    imsave("reconstr_img/org_{0}.png".format(i),      org_img)
    imsave("reconstr_img/reconstr_{0}.png".format(i), reconstr_img)


def disentangle_check(sess, model, manager, save_original=False):
  img = manager.get_image(shape=1, scale=2, orientation=5)
  if save_original:
    imsave("original.png", img.reshape(64, 64).astype(np.float32))
    
  batch_xs = [img]
  z_mean, z_log_sigma_sq = model.transform(sess, batch_xs)
  z_sigma_sq = np.exp(z_log_sigma_sq)[0]

  # Print variance
  zss_str = ""
  for i,zss in enumerate(z_sigma_sq):
    str = "z{0}={1:.2f}".format(i,zss)
    zss_str += str + ", "
  print(zss_str)

  # Save disentangled images
  z_m = z_mean[0]
  n_z = 10

  if not os.path.exists("disentangle_img"):
    os.mkdir("disentangle_img")

  for target_z_index in range(n_z):
    for ri in range(n_z):
      value = -3.0 + (6.0 / 9.0) * ri
      z_mean2 = np.zeros((1, n_z))
      for i in range(n_z):
        if( i == target_z_index ):
          z_mean2[0][i] = value
        else:
          z_mean2[0][i] = z_m[i]
      reconstr_img = model.generate(sess, z_mu=z_mean2)
      rimg = reconstr_img[0].reshape(64, 64)
      imsave("disentangle_img/check_z{0}_{1}.png".format(target_z_index,ri), rimg)
      

def load_checkpoints(sess):
  saver = tf.train.Saver()
  checkpoint = tf.train.get_checkpoint_state(flags.checkpoint_dir)
  if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print("loaded checkpoint: {0}".format(checkpoint.model_checkpoint_path))
  else:
    print("Could not find old checkpoint")
    if not os.path.exists(flags.checkpoint_dir):
      os.mkdir(flags.checkpoint_dir)
  return saver


def trpo_train(env, manager, saver,
              timesteps_per_batch = 1024, # what to train on
              max_kl = 0.01, 
              cg_iters = 10,
              gamma = 0.99, 
              lam = 0.98, # advantage estimation
              lr = 1e-4,
              vf_batch_size = 64,
              entcoeff=0.0,
              cg_damping=1e-2,
              vf_stepsize=3e-4,
              vf_iters =3,
              max_timesteps=0, max_episodes=0, max_iters=0,  # time constraint
              callback=None,
              save_data_freq = 100,
              save_model_freq = 100
              ):
  ob_space  = env.observation_space
  ac_space  = env.action_space
  num_control = 5 # should be changed to len(env_list)
  beta      = 4
  pi        = policy_func("pi", ob_space, ac_space, num_control, beta)
  oldpi     = policy_func("oldpi", ob_space, ac_space, num_control, beta)
  atarg     = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
  ret       = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

  ob        = U.get_placeholder_cached(name="ob")
  c_in      = U.get_placeholder_cached(name="c_in") 
  ac        = pi.pdtype.sample_placeholder([None])

  kloldnew  = oldpi.pd.kl(pi.pd)
  ent       = pi.pd.entropy()
  meankl    = U.mean(kloldnew)
  meanent   = U.mean(ent)
  entbonus  = entcoeff * meanent

  vferr     = U.mean(tf.square(pi.vpred - ret))

  ratio     = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # advantage * pnew / pold
  surrgain  = U.mean(ratio * atarg)

  optimgain = surrgain + entbonus
  latent_loss = pi.latent_loss
  losses    = [optimgain, meankl, entbonus, surrgain, meanent, latent_loss]
  loss_names = ["optimgain", "meankl", "entloss", "surrgain", "entropy", "latent_loss"]

  dist      = meankl

  all_var_list  = pi.get_trainable_variables()
  
  var_list      = [v for v in all_var_list if v.name.split("/")[1].startswith("pol")]
  vf_var_list   = [v for v in all_var_list if v.name.split("/")[1].startswith("vf")]
  enc_var_list  = [v for v in all_var_list if v.name.split("/")[1].startswith("enc")]
  task_var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("task")]

  optimizer     = tf.train.AdamOptimizer(learning_rate=lr, epsilon = 0.01/vf_batch_size)

  get_flat      = U.GetFlat(var_list)
  set_from_flat = U.SetFromFlat(var_list)
  klgrads       = tf.gradients(dist, var_list)

  flat_tangent  = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")
  shapes        = [var.get_shape().as_list() for var in var_list]
  start         = 0
  tangents      = []

  for shape in shapes:
    sz = U.intprod(shape)
    tangents.append(tf.reshape(flat_tangent[start:start+sz], shape))
    start += sz

  gvp   = tf.add_n([U.sum(g*tangent) for (g, tangent) in zipsame(klgrads, tangents)]) #pylint: disable=E1111
  fvp   = U.flatgrad(gvp, var_list)

  assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
    for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])

  compute_losses = U.function([ob, ac, atarg], losses)
  compute_lossandgrad = U.function([ob, ac, atarg], losses + [U.flatgrad(optimgain, var_list)])
  compute_fvp = U.function([flat_tangent, ob, ac, atarg], fvp)
  compute_vflossandgrad = U.function([ob, ret], U.flatgrad(vferr, vf_var_list))

  vf_optimize_expr = optimizer.minimize(vferr, var_list=vf_var_list)
  vf_train = U.function([ob, ret], vferr, updates = [vf_optimize_expr])


  U.initialize()
  th_init = get_flat()
  set_from_flat(th_init)

  seg_gen = traj_segment_generator(pi, env, timesteps_per_batch, stochastic=True)

  episodes_so_far = 0
  timesteps_so_far = 0
  iters_so_far = 0

  assert sum([max_iters>0, max_timesteps>0, max_episodes>0])==1

  iter_log = []
  epis_log = []
  timestep_log = []
  ret_mean_log = []
  ret_std_log = []

  saver = tf.train.Saver()
  meta_saved = False


  while True:        
    if callback: callback(locals(), globals())
    if max_timesteps and timesteps_so_far >= max_timesteps:
      print "Max Timestep : {}".format(timesteps_so_far)
      break
    elif max_episodes and episodes_so_far >= max_episodes:
      print "Max Episodes : {}".format(episodes_so_far)
      break
    elif max_iters and iters_so_far >= max_iters:
      print "Max Iter : {}".format(iters_so_far)
      break
    warn("********** Iteration %i ************"%iters_so_far)

    # with timed("sampling"):
    #       seg = seg_gen.__next__()
    seg = traj_segment_generator(pi, env, timesteps_per_batch, stochastic=True)
    add_vtarg_and_adv(seg, gamma, lam)

    # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
    ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
    vpredbefore = seg["vpred"] # predicted value function before udpate
    atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate

    if hasattr(pi, "ret_rms"): pi.ret_rms.update(tdlamret)
    if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

    args = seg["ob"], seg["ac"], seg["adv"]
    fvpargs = [arr[::5] for arr in args]
    def fisher_vector_product(p):
      return compute_fvp(p, *fvpargs) + cg_damping * p

    assign_old_eq_new()
    surrbefore, _,_,_,_, g = compute_lossandgrad(*args)
    surrbefore = np.array(surrbefore)
    if np.allclose(g, 0):
      print("Got zero gradient. not updating")
    else:
      stepdir = U.conjugate_gradient(fisher_vector_product, g, cg_iters=cg_iters)
      assert np.isfinite(stepdir).all()
      shs = .5*stepdir.dot(fisher_vector_product(stepdir))
      lm = np.sqrt(shs / max_kl)
      fullstep = stepdir / lm
      expectedimprove = g.dot(fullstep)
      stepsize = 1.0
      thbefore = get_flat()
      for _ in range(10):
        thnew = thbefore + fullstep * stepsize
        set_from_flat(thnew)
        meanlosses = surr, kl, _,_,_ = np.array(compute_losses(*args))
        improve = surr - surrbefore
        if not np.isfinite(meanlosses).all():
          print("Got non-finite value of losses -- bad!")
        elif kl > max_kl * 1.5:
          print("violated KL constraint. shrinking step.")
        elif improve < 0:
          print("surrogate didn't improve. shrinking step.")
        else:
          break
          stepsize *= .5
      else:
        print("couldn't compute a good step")
        set_from_flat(thbefore)

    for _ in range(vf_iters):
      for (mbob, mbret) in dataset.iterbatches((seg["ob"], seg["tdlamret"]), 
        include_final_partial_batch=False, batch_size=64):
        vfloss = vf_train(mbob, mbret)


    episodes_so_far += len(seg["ep_lens"])
    timesteps_so_far += sum(seg["ep_lens"])
    iters_so_far += 1

    mean_ret = np.mean(seg["ep_rets"])
    std_ret = np.std(seg["ep_rets"])


    iter_log.append(iters_so_far)
    epis_log.append(episodes_so_far)
    timestep_log.append(timesteps_so_far)
    ret_mean_log.append(mean_ret)
    ret_std_log.append(std_ret)

    if iters_so_far % save_model_freq == 1:
      if meta_saved == True:
        saver.save(U.get_session(), 'my_test_model', global_step = iters_so_far, write_meta_graph = False)
      else:
        print "Save  meta graph"
        saver.save(U.get_session(), 'my_test_model', global_step = iters_so_far, write_meta_graph = True)
        meta_saved = True


    if iters_so_far % save_data_freq == 1:
      iter_log_d = pd.DataFrame(iter_log)
      epis_log_d = pd.DataFrame(epis_log)
      timestep_log_d = pd.DataFrame(timestep_log)
      ret_mean_log_d = pd.DataFrame(ret_mean_log)
      ret_std_log_d = pd.DataFrame(ret_std_log)

      save_file = "test_iter_{}.h5".format(iters_so_far)
      with pd.HDFStore(save_file, 'w') as outf:
        outf['iter_log'] = iter_log_d
        outf['epis_log'] = epis_log_d
        outf['timestep_log'] = timestep_log_d
        outf['ret_mean_log'] = ret_mean_log_d
        outf['ret_std_log'] = ret_std_log_d
        
        filesave('Wrote {}'.format(save_file))

    header('iters_so_far : {}'.format(iters_so_far))
    header('timesteps_so_far : {}'.format(timesteps_so_far))
    header('mean_ret : {}'.format(mean_ret))
    header('std_ret : {}'.format(std_ret))


def main(env_id):

  import tf_util as U

  sess = U.single_threaded_session()
  sess.__enter__()

  env = gym.make(env_id)

  print(env.observation_space.shape)

  def policy_fn(name, ob_space, ac_space, num_control, beta):
        return LatentRL(name=name, ob_space=env.observation_space, ac_space=env.action_space, num_control=num_control, beta = beta)


  saver = load_checkpoints(sess)

  # if flags.training:
  #   # Train
  trpo_train(env, policy_fn, manager, saver)
  # else:
  #   reconstruct_check_images = manager.get_random_images(10)
  #   # Image reconstruction check
  #   reconstruct_check(sess, model, reconstruct_check_images)
  #   # Disentangle check
  #   disentangle_check(sess, model, manager)
  

if __name__ == '__main__':
  env_id = "Hopper-v1"
  main(env_id= env_id)
