import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;
import 'package:mobile_scanner/mobile_scanner.dart';
import 'package:model_viewer_plus/model_viewer_plus.dart';
import 'package:qr_flutter/qr_flutter.dart';

import 'chat_session.dart';

enum ChatWorkspaceMode { textToSign, signToText }

class ChatWorkspaceConfig {
  const ChatWorkspaceConfig({required this.mode});

  final ChatWorkspaceMode mode;
}

class ChatPage extends StatefulWidget {
  const ChatPage({super.key, required this.apiBaseUrl});

  final String apiBaseUrl;

  @override
  State<ChatPage> createState() => _ChatPageState();
}

class _ChatPageState extends State<ChatPage> {
  final ChatSession _session = ChatSession.instance;
  final TextEditingController _roomController = TextEditingController();
  final ScrollController _scrollController = ScrollController();

  @override
  void initState() {
    super.initState();
    _session.addListener(_onSessionChanged);
    if (_session.isActive) _session.setChatVisible(true);
  }

  @override
  void dispose() {
    _session.removeListener(_onSessionChanged);
    _session.setChatVisible(false);
    _roomController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  void _onSessionChanged() {
    if (!mounted) return;
    if (_session.isActive) _session.setChatVisible(true);
    setState(() {});
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 220),
          curve: Curves.easeOut,
        );
      }
    });
  }

  Future<void> _run(Future<void> Function() action) async {
    try {
      await action();
    } catch (error) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(error.toString().replaceFirst('Exception: ', '')),
        ),
      );
    }
  }

  Future<void> _scanInvitation() async {
    final value = await Navigator.of(
      context,
    ).push<String>(MaterialPageRoute(builder: (_) => const _QrScannerPage()));
    if (value != null) {
      await _run(() => _session.joinInvitation(widget.apiBaseUrl, value));
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('建立聊天室'),
        actions: [
          if (_session.isActive)
            IconButton(
              tooltip: '離開房間',
              onPressed: () => _run(_session.leave),
              icon: const Icon(Icons.logout_rounded),
            ),
        ],
      ),
      body: SafeArea(
        child: _session.isActive ? _buildModeChooser() : _buildPairing(),
      ),
    );
  }

  Widget _buildPairing() {
    return ListView(
      padding: const EdgeInsets.all(20),
      children: [
        const Icon(Icons.connect_without_contact_rounded, size: 72),
        const SizedBox(height: 16),
        const Text(
          '選擇對話方式',
          textAlign: TextAlign.center,
          style: TextStyle(fontSize: 24, fontWeight: FontWeight.w700),
        ),
        const SizedBox(height: 28),
        FilledButton.icon(
          onPressed: _session.connecting
              ? null
              : () => _run(() => _session.create(widget.apiBaseUrl)),
          icon: const Icon(Icons.add_link_rounded),
          label: const Text('建立臨時房間'),
        ),
        const Padding(
          padding: EdgeInsets.symmetric(vertical: 18),
          child: Row(
            children: [
              Expanded(child: Divider()),
              Padding(
                padding: EdgeInsets.symmetric(horizontal: 12),
                child: Text('或加入房間'),
              ),
              Expanded(child: Divider()),
            ],
          ),
        ),
        OutlinedButton.icon(
          onPressed: _session.connecting ? null : _scanInvitation,
          icon: const Icon(Icons.qr_code_scanner_rounded),
          label: const Text('掃描 QR Code'),
        ),
        const SizedBox(height: 12),
        TextField(
          controller: _roomController,
          textCapitalization: TextCapitalization.characters,
          maxLength: 8,
          decoration: InputDecoration(
            labelText: '房間碼',
            hintText: '例如 AB3D7K9M',
            border: const OutlineInputBorder(),
            suffixIcon: IconButton(
              tooltip: '加入',
              onPressed: _session.connecting
                  ? null
                  : () => _run(
                      () => _session.join(
                        widget.apiBaseUrl,
                        _roomController.text,
                      ),
                    ),
              icon: const Icon(Icons.arrow_forward_rounded),
            ),
          ),
          onSubmitted: _session.connecting
              ? null
              : (value) => _run(() => _session.join(widget.apiBaseUrl, value)),
        ),
        if (_session.connecting)
          const Padding(
            padding: EdgeInsets.only(top: 12),
            child: Center(child: CircularProgressIndicator()),
          ),
      ],
    );
  }

  Widget _buildModeChooser() {
    final invitation = _session.invitationUri;
    return ListView(
      padding: const EdgeInsets.all(20),
      children: [
        Material(
          color: const Color(0xFFEAF1FF),
          borderRadius: BorderRadius.circular(16),
          child: Padding(
            padding: const EdgeInsets.fromLTRB(16, 10, 12, 10),
            child: Row(
              children: [
                Icon(
                  _session.onlineCount >= 2
                      ? Icons.check_circle_rounded
                      : Icons.hourglass_top_rounded,
                  color: _session.onlineCount >= 2
                      ? Colors.green
                      : Colors.orange,
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    _session.localMode
                        ? '單機對話'
                        : _session.onlineCount >= 2
                        ? '對方已連線 · 房間 ${_session.roomId}'
                        : '等待對方加入 · 房間 ${_session.roomId}',
                  ),
                ),
                if (invitation != null)
                  IconButton(
                    tooltip: '顯示邀請 QR Code',
                    onPressed: () => _showInvitation(invitation),
                    icon: const Icon(Icons.qr_code_2_rounded),
                  ),
              ],
            ),
          ),
        ),
        const SizedBox(height: 32),
        const Text(
          '選擇翻譯模式',
          style: TextStyle(fontSize: 22, fontWeight: FontWeight.w700),
        ),
        const SizedBox(height: 8),
        const Text('進入後可拖曳移動聊天室大小', style: TextStyle(color: Colors.black54)),
        const SizedBox(height: 24),
        FilledButton.icon(
          style: FilledButton.styleFrom(minimumSize: const Size.fromHeight(64)),
          onPressed: () => Navigator.of(
            context,
          ).pop(const ChatWorkspaceConfig(mode: ChatWorkspaceMode.textToSign)),
          icon: const Icon(Icons.translate_rounded),
          label: const Text('文字轉手語'),
        ),
        const SizedBox(height: 14),
        OutlinedButton.icon(
          style: OutlinedButton.styleFrom(
            minimumSize: const Size.fromHeight(64),
          ),
          onPressed: () => Navigator.of(
            context,
          ).pop(const ChatWorkspaceConfig(mode: ChatWorkspaceMode.signToText)),
          icon: const Icon(Icons.sign_language_rounded),
          label: const Text('手語轉文字'),
        ),
      ],
    );
  }

  Future<void> _showInvitation(String invitation) async {
    await showDialog<void>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('掃描加入房間'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            QrImageView(data: invitation, size: 230),
            const SizedBox(height: 12),
            const Text('房間碼'),
            SelectableText(
              _session.roomId!,
              style: const TextStyle(fontSize: 24, fontWeight: FontWeight.w700),
            ),
          ],
        ),
        actions: [
          TextButton.icon(
            onPressed: () {
              Clipboard.setData(ClipboardData(text: _session.roomId!));
              Navigator.pop(context);
            },
            icon: const Icon(Icons.copy_rounded),
            label: const Text('複製房間碼'),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('完成'),
          ),
        ],
      ),
    );
  }
}

class ChatSignThumbnail extends StatefulWidget {
  const ChatSignThumbnail({
    super.key,
    required this.apiBaseUrl,
    required this.message,
  });

  final String apiBaseUrl;
  final ChatMessage message;

  @override
  State<ChatSignThumbnail> createState() => _ChatSignThumbnailState();
}

class _ChatSignThumbnailState extends State<ChatSignThumbnail> {
  String? _modelUrl;
  String? _idleModelUrl;
  String? _error;
  File? _modelFile;
  File? _idleModelFile;
  bool _playing = true;

  @override
  void initState() {
    super.initState();
    unawaited(_loadAnimation());
  }

  Uri _apiUri(String path) {
    final base = widget.apiBaseUrl.replaceFirst(RegExp(r'/+$'), '');
    return Uri.parse(path.startsWith('/') ? '$base$path' : '$base/$path');
  }

  Future<void> _loadAnimation() async {
    try {
      var animationPath = widget.message.animationUrl;
      if (animationPath.isEmpty) {
        final response = await http
            .post(
              _apiUri('/api/v1/text-to-sign'),
              headers: const {
                'Content-Type': 'application/json; charset=utf-8',
                'ngrok-skip-browser-warning': 'true',
              },
              body: jsonEncode({'text': widget.message.tsl}),
            )
            .timeout(const Duration(minutes: 2));
        if (response.statusCode != 200) {
          throw HttpException('無法建立手語動畫 (${response.statusCode})');
        }
        final data = jsonDecode(utf8.decode(response.bodyBytes));
        animationPath = data['glb_url']?.toString() ?? '';
      }
      if (animationPath.isEmpty) throw const FormatException('訊息沒有手語動畫');

      final animationUri = Uri.tryParse(animationPath)?.hasScheme == true
          ? Uri.parse(animationPath)
          : _apiUri(animationPath);
      final response = await http
          .get(
            animationUri,
            headers: const {'ngrok-skip-browser-warning': 'true'},
          )
          .timeout(const Duration(minutes: 2));
      if (response.statusCode != 200 || response.bodyBytes.length < 4) {
        throw HttpException('動畫下載失敗 (${response.statusCode})');
      }
      final signature = ascii.decode(
        response.bodyBytes.sublist(0, 4),
        allowInvalid: true,
      );
      if (signature != 'glTF') throw const FormatException('動畫檔案格式錯誤');

      final file = File(
        '${Directory.systemTemp.path}${Platform.pathSeparator}'
        'chat_thumbnail_${DateTime.now().microsecondsSinceEpoch}.glb',
      );
      await file.writeAsBytes(response.bodyBytes, flush: true);
      final idleResponse = await http
          .get(
            _apiUri('/actions/idle.glb'),
            headers: const {'ngrok-skip-browser-warning': 'true'},
          )
          .timeout(const Duration(minutes: 2));
      if (idleResponse.statusCode != 200 || idleResponse.bodyBytes.length < 4) {
        throw HttpException('待機動畫下載失敗 (${idleResponse.statusCode})');
      }
      final idleFile = File(
        '${Directory.systemTemp.path}${Platform.pathSeparator}'
        'chat_thumbnail_idle_${DateTime.now().microsecondsSinceEpoch}.glb',
      );
      await idleFile.writeAsBytes(idleResponse.bodyBytes, flush: true);
      if (!mounted) {
        await file.delete();
        await idleFile.delete();
        return;
      }
      setState(() {
        _modelFile = file;
        _modelUrl = file.uri.toString();
        _idleModelFile = idleFile;
        _idleModelUrl = idleFile.uri.toString();
      });
    } catch (error) {
      if (mounted) setState(() => _error = error.toString());
    }
  }

  Future<void> _togglePlayback() async {
    if (_playing) {
      if (mounted) setState(() => _playing = false);
      return;
    }
    setState(() => _playing = true);
  }

  void _openFullScreen() {
    Navigator.of(context).push(
      MaterialPageRoute(
        builder: (_) => FullScreenSignPlayer(
          apiBaseUrl: widget.apiBaseUrl,
          message: widget.message,
        ),
      ),
    );
  }

  @override
  void dispose() {
    final file = _modelFile;
    if (file != null) unawaited(file.delete().catchError((_) => file));
    final idleFile = _idleModelFile;
    if (idleFile != null) {
      unawaited(idleFile.delete().catchError((_) => idleFile));
    }
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(12),
      child: SizedBox(
        width: 220,
        child: AspectRatio(
          key: ValueKey('chat-animation-thumbnail-${widget.message.id}'),
          aspectRatio: 3 / 4,
          child: Stack(
            fit: StackFit.expand,
            children: [
              ColoredBox(
                color: const Color(0xFFC6C6C6),
                child: _modelUrl != null && _idleModelUrl != null
                    ? ModelViewer(
                        key: ValueKey(_playing ? _modelUrl : _idleModelUrl),
                        src: _playing ? _modelUrl! : _idleModelUrl!,
                        alt: '聊天室手語動畫縮圖',
                        backgroundColor: const Color(0xFFC6C6C6),
                        // Start the sign animation immediately when the GLB is
                        // ready. The JS listener still limits it to one pass.
                        autoPlay: true,
                        cameraControls: false,
                        cameraOrbit: '0deg 75deg 45%',
                        cameraTarget: '0m 1.05m 0m',
                        loading: Loading.eager,
                        javascriptChannels: _playing
                            ? {
                                JavascriptChannel(
                                  'ThumbnailAnimation',
                                  onMessageReceived: (_) {
                                    if (mounted) {
                                      setState(() => _playing = false);
                                    }
                                  },
                                ),
                              }
                            : null,
                        relatedJs: _playing
                            ? '''
const viewer = document.querySelector('model-viewer');
let completionReported = false;
viewer.addEventListener('finished', () => {
  if (completionReported) return;
  completionReported = true;
  ThumbnailAnimation.postMessage('finished');
});
viewer.addEventListener('load', () => {
  completionReported = false;
  viewer.currentTime = 0;
  viewer.play({repetitions: 1});
}, {once: true});
'''
                            : null,
                      )
                    : Center(
                        child: _error == null
                            ? const CircularProgressIndicator()
                            : const Icon(Icons.broken_image_outlined, size: 36),
                      ),
              ),
              Material(
                color: Colors.transparent,
                child: InkWell(
                  key: ValueKey('open-chat-animation-${widget.message.id}'),
                  onTap: _openFullScreen,
                ),
              ),
              Positioned(
                right: 8,
                bottom: 8,
                child: IconButton.filled(
                  key: ValueKey('toggle-chat-animation-${widget.message.id}'),
                  tooltip: _playing ? '暫停動畫' : '播放動畫',
                  onPressed: _togglePlayback,
                  icon: Icon(
                    _playing ? Icons.pause_rounded : Icons.play_arrow_rounded,
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class FullScreenSignPlayer extends StatefulWidget {
  const FullScreenSignPlayer({
    super.key,
    required this.apiBaseUrl,
    required this.message,
  });

  final String apiBaseUrl;
  final ChatMessage message;

  @override
  State<FullScreenSignPlayer> createState() => _FullScreenSignPlayerState();
}

class _FullScreenSignPlayerState extends State<FullScreenSignPlayer> {
  String? _modelUrl;
  String? _idleModelUrl;
  String? _error;
  File? _modelFile;
  File? _idleModelFile;
  bool _playing = true;

  @override
  void initState() {
    super.initState();
    unawaited(_loadAnimation());
  }

  Uri _apiUri(String path) {
    final base = widget.apiBaseUrl.replaceFirst(RegExp(r'/+$'), '');
    return Uri.parse(path.startsWith('/') ? '$base$path' : '$base/$path');
  }

  Future<void> _loadAnimation() async {
    try {
      var animationPath = widget.message.animationUrl;
      if (animationPath.isEmpty) {
        final response = await http
            .post(
              _apiUri('/api/v1/text-to-sign'),
              headers: const {
                'Content-Type': 'application/json; charset=utf-8',
                'ngrok-skip-browser-warning': 'true',
              },
              body: jsonEncode({'text': widget.message.tsl}),
            )
            .timeout(const Duration(minutes: 2));
        if (response.statusCode != 200) {
          throw HttpException('無法建立手語動畫 (${response.statusCode})');
        }
        final data = jsonDecode(utf8.decode(response.bodyBytes));
        animationPath = data['glb_url']?.toString() ?? '';
      }
      if (animationPath.isEmpty) throw const FormatException('訊息沒有手語動畫');

      final animationUri = Uri.tryParse(animationPath)?.hasScheme == true
          ? Uri.parse(animationPath)
          : _apiUri(animationPath);
      final response = await http
          .get(
            animationUri,
            headers: const {'ngrok-skip-browser-warning': 'true'},
          )
          .timeout(const Duration(minutes: 2));
      if (response.statusCode != 200 || response.bodyBytes.length < 4) {
        throw HttpException('動畫下載失敗 (${response.statusCode})');
      }
      final signature = ascii.decode(
        response.bodyBytes.sublist(0, 4),
        allowInvalid: true,
      );
      if (signature != 'glTF') throw const FormatException('動畫檔案格式錯誤');

      final file = File(
        '${Directory.systemTemp.path}${Platform.pathSeparator}'
        'chat_sign_${DateTime.now().microsecondsSinceEpoch}.glb',
      );
      await file.writeAsBytes(response.bodyBytes, flush: true);
      final idleResponse = await http
          .get(
            _apiUri('/actions/idle.glb'),
            headers: const {'ngrok-skip-browser-warning': 'true'},
          )
          .timeout(const Duration(minutes: 2));
      if (idleResponse.statusCode != 200 || idleResponse.bodyBytes.length < 4) {
        throw HttpException('待機動畫下載失敗 (${idleResponse.statusCode})');
      }
      final idleFile = File(
        '${Directory.systemTemp.path}${Platform.pathSeparator}'
        'chat_sign_idle_${DateTime.now().microsecondsSinceEpoch}.glb',
      );
      await idleFile.writeAsBytes(idleResponse.bodyBytes, flush: true);
      if (!mounted) {
        await file.delete();
        await idleFile.delete();
        return;
      }
      setState(() {
        _modelFile = file;
        _modelUrl = file.uri.toString();
        _idleModelFile = idleFile;
        _idleModelUrl = idleFile.uri.toString();
      });
    } catch (error) {
      if (mounted) setState(() => _error = error.toString());
    }
  }

  Future<void> _togglePlayback() async {
    if (_playing) {
      if (mounted) setState(() => _playing = false);
      return;
    }
    setState(() => _playing = true);
  }

  @override
  void dispose() {
    final file = _modelFile;
    if (file != null) unawaited(file.delete().catchError((_) => file));
    final idleFile = _idleModelFile;
    if (idleFile != null) {
      unawaited(idleFile.delete().catchError((_) => idleFile));
    }
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFC6C6C6),
      body: SafeArea(
        child: Stack(
          fit: StackFit.expand,
          children: [
            if (_modelUrl != null && _idleModelUrl != null)
              ModelViewer(
                key: ValueKey(_playing ? _modelUrl : _idleModelUrl),
                src: _playing ? _modelUrl! : _idleModelUrl!,
                alt: '聊天室手語動畫',
                backgroundColor: const Color(0xFFC6C6C6),
                // Avoid showing the model's T-pose while the playback hook is
                // being attached.
                autoPlay: true,
                cameraControls: true,
                loading: Loading.eager,
                javascriptChannels: _playing
                    ? {
                        JavascriptChannel(
                          'FullscreenAnimation',
                          onMessageReceived: (_) {
                            if (mounted) setState(() => _playing = false);
                          },
                        ),
                      }
                    : null,
                relatedJs: _playing
                    ? '''
const viewer = document.querySelector('model-viewer');
let completionReported = false;
viewer.addEventListener('finished', () => {
  if (completionReported) return;
  completionReported = true;
  FullscreenAnimation.postMessage('finished');
});
viewer.addEventListener('load', () => {
  completionReported = false;
  viewer.currentTime = 0;
  viewer.play({repetitions: 1});
}, {once: true});
'''
                    : null,
              )
            else if (_error != null)
              Center(
                child: Padding(
                  padding: const EdgeInsets.all(32),
                  child: Text('無法播放動畫：$_error', textAlign: TextAlign.center),
                ),
              )
            else
              const Center(child: CircularProgressIndicator()),
            Positioned(
              left: 14,
              top: 14,
              child: IconButton.filled(
                tooltip: '關閉全螢幕動畫',
                onPressed: () => Navigator.of(context).pop(),
                icon: const Icon(Icons.close_rounded),
              ),
            ),
            Positioned(
              right: 14,
              top: 14,
              child: IconButton.filled(
                tooltip: _playing ? '暫停動畫' : '播放動畫',
                onPressed: _togglePlayback,
                icon: Icon(
                  _playing ? Icons.pause_rounded : Icons.play_arrow_rounded,
                ),
              ),
            ),
            Positioned(
              left: 24,
              right: 24,
              bottom: 24,
              child: Text(
                widget.message.text,
                textAlign: TextAlign.center,
                style: const TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.w600,
                  color: Colors.black87,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _QrScannerPage extends StatefulWidget {
  const _QrScannerPage();

  @override
  State<_QrScannerPage> createState() => _QrScannerPageState();
}

class _QrScannerPageState extends State<_QrScannerPage> {
  bool _handled = false;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('掃描聊天室 QR Code')),
      body: MobileScanner(
        onDetect: (capture) {
          if (_handled) return;
          final value = capture.barcodes.firstOrNull?.rawValue;
          if (value == null || !value.startsWith('tslchat://join')) return;
          _handled = true;
          Navigator.of(context).pop(value);
        },
      ),
    );
  }
}
