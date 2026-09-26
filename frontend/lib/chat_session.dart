import 'dart:async';
import 'dart:convert';
import 'dart:math';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:web_socket_channel/web_socket_channel.dart';

class ChatMessage {
  const ChatMessage({
    required this.id,
    required this.senderId,
    required this.text,
    required this.source,
    required this.createdAt,
    this.tsl = '',
    this.animationUrl = '',
  });

  final String id;
  final String senderId;
  final String text;
  final String source;
  final String tsl;
  final String animationUrl;
  final DateTime createdAt;

  factory ChatMessage.fromJson(Map<String, dynamic> json) => ChatMessage(
    id: json['id']?.toString() ?? '',
    senderId: json['sender_id']?.toString() ?? '',
    text: json['text']?.toString() ?? '',
    source: json['source']?.toString() ?? 'text',
    tsl: json['tsl']?.toString() ?? '',
    animationUrl: json['animation_url']?.toString() ?? '',
    createdAt: DateTime.fromMillisecondsSinceEpoch(
      (json['created_at'] as num?)?.toInt() ?? 0,
    ),
  );
}

class ChatSession extends ChangeNotifier {
  ChatSession._();

  static final ChatSession instance = ChatSession._();

  final String deviceId = _randomToken(16);
  final List<ChatMessage> messages = [];
  WebSocketChannel? _channel;
  StreamSubscription<dynamic>? _subscription;
  String? roomId;
  String? joinToken;
  String? memberToken;
  String? error;
  int onlineCount = 0;
  bool connecting = false;
  bool localMode = false;
  bool _chatVisible = false;
  int unreadCount = 0;

  bool get isConnected => _channel != null && roomId != null;
  bool get isActive => localMode || isConnected;

  String? get invitationUri {
    if (roomId == null || joinToken == null) return null;
    return Uri(
      scheme: 'tslchat',
      host: 'join',
      queryParameters: {'room': roomId!, 'token': joinToken!},
    ).toString();
  }

  static String _randomToken(int byteCount) {
    final random = Random.secure();
    return List<int>.generate(
      byteCount,
      (_) => random.nextInt(256),
    ).map((value) => value.toRadixString(16).padLeft(2, '0')).join();
  }

  Future<void> create(String apiBaseUrl) async {
    localMode = false;
    await _requestMembership(apiBaseUrl, '/api/v1/chat/rooms', {
      'device_id': deviceId,
    });
  }

  Future<void> join(
    String apiBaseUrl,
    String room, {
    String? invitationToken,
  }) async {
    localMode = false;
    final body = <String, dynamic>{
      'device_id': deviceId,
      'room_id': room.trim().toUpperCase(),
    };
    if (invitationToken != null) body['join_token'] = invitationToken;
    await _requestMembership(apiBaseUrl, '/api/v1/chat/rooms/join', body);
  }

  void startLocal() {
    localMode = true;
    roomId = null;
    joinToken = null;
    memberToken = null;
    onlineCount = 1;
    messages.clear();
    error = null;
    unreadCount = 0;
    notifyListeners();
  }

  Future<void> joinInvitation(String apiBaseUrl, String invitation) async {
    final uri = Uri.tryParse(invitation.trim());
    if (uri == null || uri.scheme != 'tslchat' || uri.host != 'join') {
      throw const FormatException('這不是有效的聊天室邀請碼');
    }
    final room = uri.queryParameters['room'];
    final token = uri.queryParameters['token'];
    if (room == null || token == null) {
      throw const FormatException('聊天室邀請碼不完整');
    }
    await join(apiBaseUrl, room, invitationToken: token);
  }

  Future<void> _requestMembership(
    String apiBaseUrl,
    String path,
    Map<String, dynamic> body,
  ) async {
    connecting = true;
    error = null;
    notifyListeners();
    try {
      final response = await http
          .post(
            Uri.parse('${apiBaseUrl.replaceFirst(RegExp(r'/+$'), '')}$path'),
            headers: const {'Content-Type': 'application/json; charset=utf-8'},
            body: jsonEncode(body),
          )
          .timeout(const Duration(seconds: 15));
      final data = jsonDecode(utf8.decode(response.bodyBytes));
      if (response.statusCode < 200 || response.statusCode >= 300) {
        throw Exception(data is Map ? data['detail'] ?? '無法加入聊天室' : '無法加入聊天室');
      }
      final result = data as Map<String, dynamic>;
      roomId = result['room_id'] as String;
      joinToken = result['join_token'] as String;
      memberToken = result['member_token'] as String;
      messages.clear();
      unreadCount = 0;
      await _connectSocket(apiBaseUrl);
    } catch (exception) {
      error = exception.toString().replaceFirst('Exception: ', '');
      rethrow;
    } finally {
      connecting = false;
      notifyListeners();
    }
  }

  Future<void> _connectSocket(String apiBaseUrl) async {
    await _subscription?.cancel();
    await _channel?.sink.close();
    final base = Uri.parse(apiBaseUrl);
    final socketUri = base.replace(
      scheme: base.scheme == 'https' ? 'wss' : 'ws',
      path: '/api/v1/chat/ws/$roomId',
      queryParameters: {'device_id': deviceId, 'member_token': memberToken!},
    );
    final channel = WebSocketChannel.connect(socketUri);
    await channel.ready.timeout(const Duration(seconds: 15));
    _channel = channel;
    _subscription = channel.stream.listen(
      _handleSocketEvent,
      onError: (Object exception) {
        error = '聊天室連線中斷：$exception';
        _channel = null;
        onlineCount = 0;
        notifyListeners();
      },
      onDone: () {
        _channel = null;
        onlineCount = 0;
        notifyListeners();
      },
    );
  }

  void _handleSocketEvent(dynamic event) {
    final data = jsonDecode(event as String) as Map<String, dynamic>;
    switch (data['type']) {
      case 'history':
        messages
          ..clear()
          ..addAll(
            (data['messages'] as List<dynamic>? ?? const [])
                .whereType<Map<String, dynamic>>()
                .map(ChatMessage.fromJson),
          );
      case 'message':
        final message = ChatMessage.fromJson(data);
        if (!messages.any((item) => item.id == message.id)) {
          messages.add(message);
          if (!_chatVisible && message.senderId != deviceId) unreadCount++;
        }
      case 'presence':
        onlineCount = (data['member_count'] as num?)?.toInt() ?? 0;
      case 'error':
        error = data['message']?.toString();
    }
    notifyListeners();
  }

  void setChatVisible(bool visible) {
    if (_chatVisible == visible && (!visible || unreadCount == 0)) return;
    _chatVisible = visible;
    if (visible) unreadCount = 0;
    notifyListeners();
  }

  void send({
    required String text,
    String source = 'text',
    String tsl = '',
    String animationUrl = '',
  }) {
    if (!isActive || text.trim().isEmpty) return;
    if (localMode) {
      messages.add(
        ChatMessage(
          id: _randomToken(12),
          senderId: deviceId,
          text: text.trim(),
          source: source,
          tsl: tsl.trim(),
          animationUrl: animationUrl.trim(),
          createdAt: DateTime.now(),
        ),
      );
      notifyListeners();
      return;
    }
    _channel!.sink.add(
      jsonEncode({
        'type': 'message',
        'text': text.trim(),
        'source': source,
        'tsl': tsl.trim(),
        'animation_url': animationUrl.trim(),
      }),
    );
  }

  Future<void> leave() async {
    final subscription = _subscription;
    final channel = _channel;
    _subscription = null;
    _channel = null;
    roomId = null;
    joinToken = null;
    memberToken = null;
    localMode = false;
    onlineCount = 0;
    messages.clear();
    error = null;
    unreadCount = 0;
    _chatVisible = false;
    notifyListeners();
    await channel?.sink.close();
    await subscription?.cancel();
  }
}
