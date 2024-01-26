# MOTR

## 总览

### 模型搭建
```python
# 构造最终的MOTR（det+track）
model = MOTR(
    backbone, # ResNet50
    transformer, # DETR
    track_embed=query_interaction_layer, 
    num_feature_levels=args.num_feature_levels,
    num_classes=num_classes,
    num_queries=args.num_queries,
    aux_loss=args.aux_loss,
    criterion=criterion,
    with_box_refine=args.with_box_refine,
    two_stage=args.two_stage,
    memory_bank=memory_bank,
    use_checkpoint=args.use_checkpoint,
)
```

## MOTR算法流程

### BACKBONE

图片输入之后，首先经过`backbone`进行特征提取，提取后的特征还会进行`position_embedding`处理

```python

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        print("Joiner(backbone, position_embedding)")
        print("Input img size: {}".format(tensor_list.tensors.shape) )
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype)) # 调用position_embedding进行编码
        
        #torch.Size([1, 512, 100, 178])
        print("Output feature size: {}".format(out[0].tensors.shape)) 

        # torch.Size([1, 256, 100, 178])
        # 100是qury最大个数，178是特征维度
        print("Output pos size: {}".format(pos[0].shape))  
        
        return out, pos

# backbone模型定义
def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks or (args.num_feature_levels > 1)
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation) # reset
    model = Joine
```

> Input img size: torch.Size([1, 3, 800, 1422])
> 
> Output feature size: torch.Size([1, 512, 100, 178])
> 
> Output pos size: torch.Size([1, 256, 100, 178])


### DETR

模型定义
```PYTHON
def build_deforamble_transformer(args):
    return DeformableTransformer(
        d_model=args.hidden_dim, # 256
        nhead=args.nheads,      # 8
        num_encoder_layers=args.enc_layers, # 6
        num_decoder_layers=args.dec_layers, # 6
        dim_feedforward=args.dim_feedforward, # 1024
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels, # 4
        dec_n_points=args.dec_n_points, # 4
        enc_n_points=args.enc_n_points, # 4
        two_stage=args.two_stage, # False
        two_stage_num_proposals=args.num_queries, # 300
        decoder_self_cross=not args.decoder_cross_self, # False
        sigmoid_attn=args.sigmoid_attn, # False
        extra_track_attn=args.extra_track_attn, # True
    )

# 模型推理
def forward(self, srcs, masks, pos_embeds, query_embed=None, ref_pts=None):
    ...
    ...
```

### forward的输入和输出

```python
# —————————— transformer检测部分 ——————————
hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(srcs, masks, pos, 
                                                                                                track_instances.query_pos, 
                                                                                                ref_pts=track_instances.ref_pts)
```

> srcs: 前面backbone以及后续模型提取的特征
>
> masks：
>
> pos：  position_embedding
>
> **track_instances.query_pos：上一时刻的obj属性作为当前时刻的query**，这里隐性的进行目标相似度计算啊、分配啊之类的算法步骤
>
> ref_pts=track_instances.ref_pts 


后处理部分，用于解码目标类型和box

```python
# DETR后处理。，目标类型、box解码
outputs_classes = []
outputs_coords = []
for lvl in range(hs.shape[0]):
    if lvl == 0:
        reference = init_reference
    else:
        reference = inter_references[lvl - 1]
    reference = inverse_sigmoid(reference)
    outputs_class = self.class_embed[lvl](hs[lvl])  # 类型检测模块
    tmp = self.bbox_embed[lvl](hs[lvl])             # box检测模块
    if reference.shape[-1] == 4:
        tmp += reference
    else:
        assert reference.shape[-1] == 2
        tmp[..., :2] += reference
    outputs_coord = tmp.sigmoid()
    outputs_classes.append(outputs_class)
    outputs_coords.append(outputs_coord)
outputs_class = torch.stack(outputs_classes) # 类型
outputs_coord = torch.stack(outputs_coords) # box

ref_pts_all = torch.cat([init_reference[None], inter_references[:, :, :, :2]], dim=0)
out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'ref_pts': ref_pts_all[5]}
if self.aux_loss:
    out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
out['hs'] = hs[-1]
return out
```


进过上述处理后，DETR部分就结束了，后面进入航迹t`track_instances`部分


### track_instance更新

```python
def _post_process_single_image(self, frame_res, track_instances, is_last):
    with torch.no_grad():
        # 这里的pred_logits是DETR模型检测加后处理后的结果，也就是已经经过模型推理了
        if self.training:
            track_scores = frame_res['pred_logits'][0, :].sigmoid().max(dim=-1).values
        else:
            track_scores = frame_res['pred_logits'][0, :, 0].sigmoid() 
    track_instances.scores = track_scores # 更新航迹score
    
    # for i in range(len(track_scores)):
    #     print(track_instances.obj_idxes[i], track_scores[i])
    
    # 更新航迹的相关属性
    track_instances.pred_logits = frame_res['pred_logits'][0] # 检测结果类型
    track_instances.pred_boxes = frame_res['pred_boxes'][0] # 检测结果box
    track_instances.output_embedding = frame_res['hs'][0] # 检测目标的特征？
    
    if self.training:
        # the track id will be assigned by the mather.
        frame_res['track_instances'] = track_instances
        track_instances = self.criterion.match_for_single_frame(frame_res)
    else:
        # each track will be assigned an unique global id by the track base.
        # 航迹管理模块，目标的新生与删除
        self.track_base.update(track_instances)
    if self.memory_bank is not None:
        track_instances = self.memory_bank(track_instances)
        # track_instances.track_scores = track_instances.track_scores[..., 0]
        # track_instances.scores = track_instances.track_scores.sigmoid()
        if self.training:
            self.criterion.calc_loss_for_track_scores(track_instances)
            
    tmp = {}
    tmp['init_track_instances'] = self._generate_empty_tracks() # TODO ？？
    tmp['track_instances'] = track_instances 
    if not is_last:
        # track_embed：[query_interaction_layer]
        out_track_instances = self.track_embed(tmp) # embding特征更新
        frame_res['track_instances'] = out_track_instances
    else:
        frame_res['track_instances'] = None
    return frame_res
```

### QIM

模型定义：
```python
def build(args, layer_name, dim_in, hidden_dim, dim_out):
    interaction_layers = {
        'QIM': QueryInteractionModule,
    }
    assert layer_name in interaction_layers, 'invalid query interaction layer: {}'.format(layer_name)
    return interaction_layers[layer_name](args, dim_in, hidden_dim, dim_out)
```

推理部分：
```PYTHON
def _update_track_embedding(self, track_instances: Instances) -> Instances:
    if len(track_instances) == 0:
        return track_instances
    # 这里的output_embedding和query_pos随DETR步骤已经更新
    # 此处对这些feature进行处理，得到最终track_instances的query_pos和ref_pts
    
    dim = track_instances.query_pos.shape[1]
    out_embed = track_instances.output_embedding
    query_pos = track_instances.query_pos[:, :dim // 2] 
    query_feat = track_instances.query_pos[:, dim//2:]

    # 构造QKV
    q = k = query_pos + out_embed
    tgt = out_embed
    tgt2 = self.self_attn(q[:, None], k[:, None], value=tgt[:, None])[0][:, 0] # 多头注意力模型
    tgt = tgt + self.dropout1(tgt2)
    tgt = self.norm1(tgt)

    tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
    tgt = tgt + self.dropout2(tgt2)
    tgt = self.norm2(tgt)

    if self.update_query_pos:
        query_pos2 = self.linear_pos2(self.dropout_pos1(self.activation(self.linear_pos1(tgt))))
        query_pos = query_pos + self.dropout_pos2(query_pos2)
        query_pos = self.norm_pos(query_pos)
        track_instances.query_pos[:, :dim // 2] = query_pos

    query_feat2 = self.linear_feat2(self.dropout_feat1(self.activation(self.linear_feat1(tgt))))
    query_feat = query_feat + self.dropout_feat2(query_feat2)
    query_feat = self.norm_feat(query_feat)
    track_instances.query_pos[:, dim//2:] = query_feat # 此处更新目标query_pos的feature

    track_instances.ref_pts = inverse_sigmoid(track_instances.pred_boxes[:, :2].detach().clone())
    return track_instances
```


## 简单总结

MOTR整个步骤概括如下：

- Step 1：输入图片，经过Resnet网络和position_embedding编码，输出特征
  
- Step 2：上述特征经过中间层再次处理，得到新特征和编码
  
- Step 3: 执行DETR模块，除了上述的特征和pos_embed外，还需要track_instances.query_pos和ref_pts=track_instances.ref_pts
  
- Step 4：DETR模块输出hs, init_reference, inter_references用于更新track_instances：
  - track_instances.pred_logits = frame_res['pred_logits'][0] # 检测结果类型
  - track_instances.pred_boxes = frame_res['pred_boxes'][0] # 检测结果box
  - track_instances.output_embedding = frame_res['hs'][0] # detr的输出结果，

- Step 4.1： **在DTER模型中，模型会自动计算trace和det的cost并进行分配**

- Step 5：track_instances更新
  
- Step 5.1：目标的新建与删除，这里主要依据目标的score进行判定

- Step 5.2: track_instances属性更新，使用一个多头注意力模块进行特征更新 **(QueryInteractionModule)**，更新的属性是track_instances.query_pos和track_instances.ref_pts


## 参考资料
- [MOTR解读](https://zhuanlan.zhihu.com/p/373339737)
  
- [多目标追踪——【Transformer】MOTR: End-to-End Multiple-Object Tracking with TRansformer](https://blog.csdn.net/qq_42312574/article/details/127625903)